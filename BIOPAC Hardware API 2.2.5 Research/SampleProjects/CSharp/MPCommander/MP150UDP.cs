/*
Copyright 2005-2023 BIOPAC Systems, Inc.

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

Portions Copyright 2005-2023 BIOPAC Systems, Inc.

2. Altered source versions must be plainly marked as such, and must not be 
misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
*/

using System;
using System.Windows.Forms;

using MPDEV = Biopac.API.MPDevice.MPDevImports;
using MPCODE = Biopac.API.MPDevice.MPDevImports.MPRETURNCODE;

namespace MPCommander
{
	/// <summary>
	/// Description of MP150UDP.
	/// </summary>
	public class MP150UDP : MPDevice
	{
		public MP150UDP()
		{
		}
		
		public override bool Connect(string SN)
		{
			//validate serial number
			if(SN.Trim() == "")
			{
				Console.WriteLine("Invalid MP160 serial number.");
				
				return false;
			}
			
			MPCODE retval;
			
			retval = MPDEV.connectMPDev(MPDEV.MPTYPE.MP160, MPDEV.MPCOMTYPE.MPUDP, SN);
			
			if(retval == MPCODE.MPNOTINNET)
			{
				Console.WriteLine("MP160 with serial number '{0}' is not in the network.", SN);
				return false;
			}
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Unable to connect to an MP160.");
				return false;
			}
			
			return true;
		}
		
		public override bool Configure()
		{
			MPCODE retval;
			
			//set frequency to 200 Hz
			retval = MPDEV.setSampleRate(1000.0/200.0);
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to configure the MP160.");
				return false;
			}
			
			//set to acquire on all digital and analog channels
			bool[] analog = new bool[16];
			
			for(int i=0; i<analog.Length; i++)
				analog[i] = true;
			
			retval = MPDEV.setAcqChannels(analog);
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to configure the MP160.");
				return false;
			}
			
			bool[] digital = new bool[16];
			
			for(int i=0; i<digital.Length; i++)
				digital[i] = true;
			
			retval = MPDEV.setDigitalAcqChannels(digital);
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to configure the MP160.");
				return false;
			}
			
			//load preset file into memory
			string presetfilepath = Application.StartupPath + @"\resource\channelpresets.xml";
			
			retval  = MPDEV.loadXMLPresetFile(presetfilepath);
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to load preset file.");
				return false;
			}
			
			
			return true;
		}
		
		public override bool IsConnected()
		{
			MPCODE retval;
			
			//retval = MPSUCESS implies that the MP Device is ready
			retval = MPDEV.getStatusMPDev();
			
			return (retval != MPCODE.MPSUCCESS);
		}
		
		public override bool SetPreset(string uid, int ch)
		{
			MPCODE retval = MPCODE.MPSUCCESS;
			
			if(ch < 0 || ch > 15)
			{
				Console.WriteLine("Invalid channel {0}", ch);
				Console.WriteLine("Channels are from 0 to 15.", ch);
				return false;
			}
			
			
			retval = MPDEV.configChannelByPresetID((uint) ch, uid);
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Invalid preset id '{0}'", uid);
				Console.WriteLine("Please see 'channelpresets.xml' for a valid preset id");
				return false;
			}
			
			Console.WriteLine("Channel {0} has been configured with preset {1}", ch, uid);
			
			return true;				
		}
		
		public override bool SetAllDigitalOutputChannels(bool level)
		{
			MPCODE retval = MPCODE.MPSUCCESS;
			
			//set low bits in memory first
			for(int i=0; i<8; i++)
			{
				retval = MPDEV.setDigitalIO((uint) i, level, false, MPDEV.DIGITALOPT.SET_LOW_BITS);
				
				if(retval != MPCODE.MPSUCCESS)
				{
					Console.WriteLine("Failed to set digital output.");
					return false;
				}
			}
			
			//set high bits in memory first
			for(int i=8; i<16; i++)
			{
				retval = MPDEV.setDigitalIO((uint) i, level, false, MPDEV.DIGITALOPT.SET_HIGH_BITS);
				
				if(retval != MPCODE.MPSUCCESS)
				{
					Console.WriteLine("Failed to set digital output.");
					return false;
				}
			}
			
			//send low bits to mp device
			retval = MPDEV.setDigitalIO((uint) 7, level, true, MPDEV.DIGITALOPT.SET_LOW_BITS);
				
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to set digital output.");
				return false;
			}
			
			//send  high bit to mp device
			retval = MPDEV.setDigitalIO((uint) 15, level, true, MPDEV.DIGITALOPT.SET_HIGH_BITS);
				
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to set digital output.");
				return false;
			}
			
			for(int ch=0; ch<16; ch++)
				Console.WriteLine("Digital Output Channel {0} is set to {1}.", ch, level.ToString().ToLower());
			
			return true;
		}
		
		public override bool SetDigitalOutputChannel(int ch, bool level)
		{
			MPCODE retval = MPCODE.MPSUCCESS;
			
			if(ch < 0 || ch > 15)
			{
				Console.WriteLine("Invalid channel {0}", ch);
				Console.WriteLine("Channels are from 0 to 15.", ch);
				return false;
			}
			
			if(ch < 8)
			{
				retval = MPDEV.setDigitalIO((uint) ch, level, true, MPDEV.DIGITALOPT.SET_LOW_BITS);
			}
			
			if(ch >= 8)
			{
				retval = MPDEV.setDigitalIO((uint) ch, level, true, MPDEV.DIGITALOPT.SET_HIGH_BITS);
			}
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to set digital output.");
				return false;
			}
			
			Console.WriteLine("Digital Output Channel {0} is set to {1}.", ch, level.ToString().ToLower());
			
			return true;
		}
		
		public override bool PrintAllDigitalChannels(bool online)
		{
			MPCODE retval = MPCODE.MPSUCCESS;
			
			if(online)
			{
				//start acquisition
				retval = MPDEV.startAcquisition();
				
				if(retval != MPCODE.MPSUCCESS)
				{
					Console.WriteLine("Failed to start acquisition.");
					return false;
				}
				
				//allocate space for all 16 analog and 16 digital channels
				double[] buff = new double[32];
				
				retval = MPDEV.getMostRecentSample(buff);
				
				if(retval != MPCODE.MPSUCCESS)
				{
					Console.WriteLine("Failed to get most recent sample.");
					return false;
				}
				
				retval = MPDEV.stopAcquisition();
				
				if(retval != MPCODE.MPSUCCESS)
				{
					Console.WriteLine("Failed to stop acquisition.");
					return false;
				}
				
				Console.WriteLine("Digital CH X = Value");
				
				//print just the digital values
				for(int i=16; i<32; i++)
					Console.WriteLine("Digital CH {0} = {1}", i-16, buff[i]);
				
			}
			else
			{
				Console.WriteLine("Digital CH X = Value");
				
				bool state = false;
				
				for(int i=0; i<8; i++)
				{
					retval = MPDEV.getDigitalIO((uint) i,out state,MPDEV.DIGITALOPT.READ_LOW_BITS);
					
					if(retval != MPCODE.MPSUCCESS)
					{
						Console.WriteLine("Failed to get digital io.");
						return false;
					}
					
					Console.WriteLine("Digital CH {0} = {1}", i, state ? 5 : 0);
				}
				
				for(int i=8; i<16; i++)
				{
					retval = MPDEV.getDigitalIO((uint) i,out state,MPDEV.DIGITALOPT.READ_HIGH_BITS);
					
					if(retval != MPCODE.MPSUCCESS)
					{
						Console.WriteLine("Failed to get digital io.");
						return false;
					}
					
					Console.WriteLine("Digital CH {0} = {1}", i, state ? 5 : 0);
				}
			}
			
			return true;
		}
		
		public override bool PrintDigitalChannel(bool online, int ch)
		{
			MPCODE retval = MPCODE.MPSUCCESS;
			
			if(ch < 0 || ch > 15)
			{
				Console.WriteLine("Invalid channel {0}", ch);
				Console.WriteLine("Channels are from 0 to 15.", ch);
				return false;
			}
			
			if(online)
			{
				//start acquisition
				retval = MPDEV.startAcquisition();
				
				if(retval != MPCODE.MPSUCCESS)
				{
					Console.WriteLine("Failed to start acquisition.");
					return false;
				}
				
				//allocate space for all 16 analog and 16 digital channels
				double[] buff = new double[32];
				
				retval = MPDEV.getMostRecentSample(buff);
				
				if(retval != MPCODE.MPSUCCESS)
				{
					Console.WriteLine("Failed to get most recent sample.");
					return false;
				}
				
				retval = MPDEV.stopAcquisition();
				
				if(retval != MPCODE.MPSUCCESS)
				{
					Console.WriteLine("Failed to stop acquisition.");
					return false;
				}
				
				Console.WriteLine("Digital CH X = Value");
				
				Console.WriteLine("Digital CH {0} = {1}", ch, buff[ch+16]);
			}
			else
			{
				bool state = false;
				
				if(ch < 8)
				{
					retval = MPDEV.getDigitalIO((uint) ch,out state,MPDEV.DIGITALOPT.READ_LOW_BITS);
					
					if(retval != MPCODE.MPSUCCESS)
					{
						Console.WriteLine("Failed to get digital io.");
						return false;
					}
				}
				
				if(ch >= 8)
				{
					retval = MPDEV.getDigitalIO((uint) ch,out state,MPDEV.DIGITALOPT.READ_HIGH_BITS);
					
					if(retval != MPCODE.MPSUCCESS)
					{
						Console.WriteLine("Failed to get digital io.");
						return false;
					}
				}
				
				Console.WriteLine("Digital CH {0} = {1}", ch, state ? 5 : 0);
			}
			
			return true;
		}
		
		public override bool SetAllAnalogOutputChannels(double voltage)
		{
			return SetAnalogOutputChannel(0,voltage) && SetAnalogOutputChannel(1,voltage);
		}
		
		public override bool SetAnalogOutputChannel(int ch, double voltage)
		{
			MPCODE retval;
			
			if(ch != 0 && ch != 1)
			{
				Console.WriteLine("Invalid channel {0}", ch);
				Console.WriteLine("Channels are from 0 to 1.", ch);
				return false;
			}
			
			if(voltage < -10.0 || ch > 10.0)
			{
				Console.WriteLine("Invalid voltage {0}", voltage);
				Console.WriteLine("Valid voltages are from -10.0 to 10.0", voltage);
				return false;
			}
			
			retval = MPDEV.setAnalogOut(voltage, ch);
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to set analog output.");
				return false;
			}
			
			Console.WriteLine("Analog Output Channel {0} is set to {1} volts.", ch, voltage);
			
			return true;
		}
		
		public override bool PrintAllAnalogChannels()
		{
			MPCODE retval;
			
			//start acquisition
			retval = MPDEV.startAcquisition();
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to start acquisition.");
				return false;
			}
			
			//allocate space for all 16 analog and 16 digital channels
			double[] buff = new double[32];
			
			retval = MPDEV.getMostRecentSample(buff);
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to get most recent sample.");
				return false;
			}
			
			retval = MPDEV.stopAcquisition();
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to stop acquisition.");
				return false;
			}
			
			Console.WriteLine("Analog CH X = Value");
			
			//print just the analog values
			for(int i=0; i<16; i++)
				Console.WriteLine("Analog CH {0} = {1}", i, buff[i]);
			
			return true;
		}
		
		public override bool PrintAnalogChannel(int ch)
		{
			MPCODE retval;
			
			if(ch < 0 || ch > 15)
			{
				Console.WriteLine("Invalid channel {0}", ch);
				Console.WriteLine("Channels are from 0 to 15.", ch);
				return false;
			}
			
			//start acquisition
			retval = MPDEV.startAcquisition();
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to start acquisition.");
				return false;
			}
			
			//allocate space for all 16 analog and 16 digital channels
			double[] buff = new double[32];
			
			retval = MPDEV.getMostRecentSample(buff);
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to get most recent sample.");
				return false;
			}
			
			retval = MPDEV.stopAcquisition();
			
			if(retval != MPCODE.MPSUCCESS)
			{
				Console.WriteLine("Failed to stop acquisition.");
				return false;
			}
			
			Console.WriteLine("Analog CH X = Value");
			
			//print just the analog values
			Console.WriteLine("Analog CH {0} = {1}", ch, buff[ch]);
			
			return true;
		}
		
		public override void Disconnect()
		{
			MPDEV.disconnectMPDev();
		}
	}
}
