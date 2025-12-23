/*
Copyright 2005-2008 BIOPAC Systems, Inc.

This software is provided 'as-is', without any express or implied warranty.
In no event will BIOPAC Systems, Inc. or BIOPAC Systems, Inc. employees be 
held liable for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it 
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not 
claim that you wrote the original software. If you use this software in a 
product, an acknowledgment (see the following) in the product documentation
is required.

Portions Copyright 2005-2008 BIOPAC Systems, Inc.

2. Altered source versions must be plainly marked as such, and must not be 
misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
*/

using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Collections;
using System.Threading;
using System.Windows.Forms;

using MP = Biopac.API.MPDevice.MPDevImports;
using MPCODE = Biopac.API.MPDevice.MPDevImports.MPRETURNCODE;

namespace Biopac.API.MPDevice
{
	/// <summary>
	/// Custom event tthat signifies the data has been received
	/// </summary>
	public class NewDataEventArgs : EventArgs
	{
		//the number of new data points
		private int count;

		//Count property
		public int Count
		{
			get
			{
				return count;
			}
		}

		//constructor
		public NewDataEventArgs(int c)
		{
			count = c;
		}
	}

	/// <summary>
	/// Delegate for handling the NewDataEvent
	/// </summary>
	public delegate void NewDataEventHandler(object sender, NewDataEventArgs e);

	/// <summary>
	/// Summary description for AcqusitionModule.
	/// </summary>
	public class AcquisitionModule
	{
		private ArrayList[] _waveforms; //arraylist of waveforms
		private Thread _acqThread; //thread for acquiring data
		private double temp; //temperature variable
		private double score; //score
		private float maxVolt; //max voltage
		private float minVolt; //min voltage
		private double sampleFreq; //in Hz

		public event NewDataEventHandler NewData;

		public AcquisitionModule(NewDataEventHandler handler, double rate)
		{
			//create 3 arraylist for the 3 waveforms
			_waveforms = new ArrayList[3];
			
			//initialize the waveforms
			for(int i = 0; i < _waveforms.Length; i++)
				_waveforms[i] = new ArrayList();

			_acqThread = null;

			temp = 0;
			score = 0;

			maxVolt = System.Single.MinValue;
			minVolt = System.Single.MaxValue;
			
			sampleFreq = rate; 

			// add the delegate
			NewData += handler;
		}

		/// <summary>
		/// Initializies the Acquisition Module
		/// </summary>
		/// <returns>false if an error is ecnoutered</returns>
		public bool initModule()
		{
			MPCODE retval;

			//connect to the MP Device
			retval = MP.connectMPDev(MP.MPTYPE.MP36,MP.MPCOMTYPE.MPUSB,"");

			if(retval != MPCODE.MPSUCCESS)
			{
				StopAcquisition();
				return false;
			}

			bool[] aCH = new bool[4];

			aCH[0] = true; //ECG (SS2L)
			aCH[1] = true; //temperature (SS6L)
			aCH[2] = true; //variable assesment (SS43L)
			aCH[3] = false;

			//set the acquisition channels
			retval = MP.setAcqChannels(aCH);

			if(retval != MPCODE.MPSUCCESS)
				return false;

			//set the sampling frequency
			retval = MP.setSampleRate(1000.0/sampleFreq);

			if(retval != MPCODE.MPSUCCESS)
				return false;

			//prompt the user to find the channel preset xml file
			OpenFileDialog presetXMLFileDialog = new OpenFileDialog();
			presetXMLFileDialog.ValidateNames = true;
            presetXMLFileDialog.Filter = "XML File (*.xml)|*.xml";
			presetXMLFileDialog.Title = "Select the Channel Presets XML File";

			string filename = "";

			do
			{
				filename = "";

				if(presetXMLFileDialog.ShowDialog() == DialogResult.OK)
					filename = presetXMLFileDialog.FileName;
				
				//load the preset file
				retval = MP.loadXMLPresetFile(filename);
			}//if the preset file is not valid prompt again
			while(retval != MPCODE.MPSUCCESS);

			//configure ECG channel (SS2L)
			retval = MP.configChannelByPresetID(0,"a102");

			if(retval != MPCODE.MPSUCCESS)
				return false;


			//configure temperature channel (SS6L)
			retval = MP.configChannelByPresetID(1,"a138");

			if(retval != MPCODE.MPSUCCESS)
				return false;

			//configure variable assesment transducer (SS43L)
			retval = MP.configChannelByPresetID(2,"a125");

			if(retval != MPCODE.MPSUCCESS)
				return false;

			MP.stopAcquisition();

			//if everything succeeds start the acquisition
			return Acquire();
		}
		
		/// <summary>
		/// Starts the acquisition
		/// </summary>
		/// <returns>false if an error is ecnoutered</returns>
		private bool Acquire()
		{
			MPCODE retval;

			//create a new thread
			_acqThread = new Thread(new ThreadStart(AcquireData));

			//start the acquisition daemon
			retval = MP.startMPAcqDaemon();

			if(retval != MPCODE.MPSUCCESS)
				return false;

			//start the acquisition
			retval = MP.startAcquisition();

			if(retval != MPCODE.MPSUCCESS)
				return false;

			//start the acquisition thread
			_acqThread.Start();

			return true;
		}

		/// <summary>
		/// Entry point for the acquisition thread
		/// </summary>
		private void AcquireData()
		{
			MPCODE retval;
			uint received = 0;

			//create a buffer that is 3 time the frequency in Hz
			//this is large enough to hold 1 second of 
			//samples from all three acquisition channels
			uint numberOfDataPoints = (uint) sampleFreq*3;
			double[] buffer = new double[numberOfDataPoints];
			

			//acquire data forever unless
			//an error occurs
			while(true)
			{
				retval = MP.receiveMPData(buffer,numberOfDataPoints, out received);

				//check for error condition
				if(retval != MPCODE.MPSUCCESS || received != numberOfDataPoints)
				{
					return;
				}

				//deinterleaved the data channels
				lock(_waveforms)
				{
					//iterate through the acquired channels
					for(int i=0; i < numberOfDataPoints; i++)
					{
						lock(_waveforms[i%3].SyncRoot)
						{
							//add the data to the correct waveform
							_waveforms[i%3].Add(buffer[i]);
							
							//if the first channel (the ecg channel)
							//contains a length that is a multiple of 50
							//raise the NewDataEvent
							if(i%3 == 0 && _waveforms[i%3].Count%50 == 0)
								//raise the event
								this.NewData(this,new NewDataEventArgs(_waveforms[i%3].Count));
						}
					}
				}
				
			}
		}

		/// <summary>
		/// Stops the acquisition and disconnects from the Hardware API
		/// </summary>
		public void StopAcquisition()
		{
			MP.stopAcquisition();
			MP.disconnectMPDev();
		}


		/// <summary>
		/// Visualize the data acquired into a graphics object
		/// This draw the following:
		/// Channel 1 (ECG) as a waveform
		/// Channel 2 and 3 as text
		/// </summary>
		/// <param name="graphics">graphics object where the data will be drawn</param>
		public void DrawData(Graphics graphics)
		{
			//initialize drawing bounds
			float maxWidth = Convert.ToSingle(graphics.VisibleClipBounds.Width);
			float maxHeight = Convert.ToSingle(graphics.VisibleClipBounds.Height);
			float waveMaxWidth = Convert.ToSingle(maxWidth * .80);
			float waveMaxHeight = maxHeight;

			//border
			PointF[] topBorder = {new PointF(0,0), new PointF(maxWidth,0)};
			PointF[] verticalBorder = {new PointF(waveMaxWidth,0), new PointF(waveMaxWidth,waveMaxHeight)};
			PointF[] horizontalBorder = {new PointF(waveMaxWidth,waveMaxHeight/2), new PointF(maxWidth,waveMaxHeight/2)};

			//pens and brushes
			Pen borderPen = new Pen(Color.White,4);
			Pen wavePen = new Pen(Color.Orange);
			Brush textBrush = Brushes.Red;
			Font textFont = new Font(FontFamily.GenericSansSerif,14);

			//offscreen graphics and bitmap
			Bitmap offscreen  = new Bitmap((int)maxWidth,(int)maxHeight);
			Graphics g = Graphics.FromImage(offscreen);
			g.Clear(Color.Black);

			//draw borders
			g.DrawLines(borderPen,topBorder);
			g.DrawLines(borderPen,verticalBorder);
			g.DrawLines(borderPen,horizontalBorder);
			
			lock(_waveforms)
			{
				//convert waveform zero to an array of points
				PointF[] waveform = WaveformToPoints(0,waveMaxWidth,waveMaxHeight);		

				//draw the points as lines on the graphics object
				if(waveform.Length > 1)
					g.DrawLines(wavePen, waveform);
				
				//draw other data as text

				//get the last sample of channel 2 (TEMPERATURE)
				lock(_waveforms[1].SyncRoot)
				{
					if(_waveforms[1].Count > 1)
						temp = (double) _waveforms[1][_waveforms[1].Count-1];
				}

				//get the last sample of channel 3 (SUBJECTIVE SCORE)
				lock(_waveforms[2].SyncRoot)
				{
					if(_waveforms[2].Count > 1)
						score = (double) _waveforms[2][_waveforms[2].Count-1];
				}
				//convert to string
				string scoreString =  "\nScore: " + score.ToString("0.00");
				string tempString =   "\nTemp: " + temp.ToString("0.00") + "º F";
				
				//draw the strings
				g.DrawString(scoreString,textFont,textBrush,new PointF(waveMaxWidth,0));
				g.DrawString(tempString,textFont,textBrush,new PointF(waveMaxWidth,waveMaxHeight/2));
			}

			//draw the offscreen graphics to the screen
			graphics.DrawImage(offscreen,0,0);
		}

		#region Auxiliary Drawing Functions
		/// <summary>
		/// Convert a waveform in to an array of points.
		/// </summary>
		/// <param name="waveIndex">The index of the waveform</param>
		/// <param name="width">the maximum width in pixels</param>
		/// <param name="height">the maximum height in pixels</param>
		/// <returns></returns>
		private PointF[] WaveformToPoints(int waveIndex, float width, float height)
		{
			float sampleRate = 1000/(float)sampleFreq;//in msec/sample
			float timeInterval = 10000; //in msec (window display time)
			float pixelInterval = (width*sampleRate)/timeInterval; //pixel/sample

			//adjust for the fact that 0 time is the first sample
			//distribute the original pixelInterval across the number of samples that fits in the time interval
			pixelInterval += pixelInterval / (timeInterval/sampleRate);

			//arraylist to store the points
			ArrayList waveCoord = new ArrayList();
			
			lock(_waveforms[waveIndex].SyncRoot)
			{
				//copy the arraylist to an array
				double[] waveform = null;
				waveform = (double[]) _waveforms[waveIndex].ToArray(typeof(double));
			

				//get the number of samples
				long samples = waveform.LongLength;
				//the number of samples to display
				long sampleLimit = (long) (timeInterval/sampleRate);
				//calculate the start sample
				long startSample = (sampleLimit > samples) ? 0 : (sampleLimit * (samples/sampleLimit))-1;
				//calculate the end sample
				long endSample = (startSample+sampleLimit >= samples) ? samples : startSample+sampleLimit;

				float t=0;
				float val = 0;

				//find the max and min
				maxVolt = System.Single.MinValue;
				minVolt = System.Single.MaxValue;

				for(long s=startSample; s<endSample; s++)
				{
					val = Convert.ToSingle(waveform[s]);
					maxVolt = (val > maxVolt ) ? val : maxVolt;
					minVolt = (val < minVolt) ? val : minVolt;
				}
				
				//convert the voltage values to points
				for(long s=startSample; s<endSample; s++,t+=pixelInterval)
				{
					val = ScaleVoltsToPixel(Convert.ToSingle(waveform[s]), Convert.ToSingle(height));
					waveCoord.Add(new PointF(t,val));
				}
			
			}

			return (PointF[]) waveCoord.ToArray(typeof(PointF));
		}

		/// <summary>
		/// Convert the voltage to Pixels values
		/// </summary>
		/// <param name="volt">the value to be converted</param>
		/// <param name="pixelHeight">the maximum height of the pixel</param>
		/// <returns>the pixel value</returns>
		private float ScaleVoltsToPixel(float volt, float pixelHeight)
		{
			float maxPixel = (pixelHeight * .10F);
			float minPixel = (pixelHeight * .90F);
									
			float m = (maxPixel-minPixel)/(maxVolt-minVolt);
			float b = maxPixel - (m*maxVolt);

			return (m*volt)+b;
		}
		#endregion

	}
}
