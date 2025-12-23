/*
Copyright 2004-2023 BIOPAC Systems, Inc.

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

Portions Copyright 2004-2023 BIOPAC Systems, Inc.

2. Altered source versions must be plainly marked as such, and must not be 
misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
*/

using MP=Biopac.API.MPDevice.MPDevImports;
using MPCODE=Biopac.API.MPDevice.MPDevImports.MPRETURNCODE;
using MPTRIG=Biopac.API.MPDevice.MPDevImports.TRIGGEROPT;
using MPDIGOPT=Biopac.API.MPDevice.MPDevImports.DIGITALOPT;

namespace Biopac.API.MPDevice.Samples
{
	/// <summary>
	/// Summary description for TempControl.
	/// </summary>
	public class TempControl
	{
		private static bool fan = false;
		private static uint[][] tensSegments;
		private static uint[][] onesSegments;
		private static uint ledCurrVal = 0;

		#region Properties
		/// <summary>
		/// The state of the Fan.
		/// True if the Analog Output 1 is set to 5 volts
		/// False if the Analog Output 1 is set to 0 volts
		/// </summary>
		public static bool Fan
		{
			get
			{
				return fan;
			}
		}

		/// <summary>
		/// The current temprature.
		/// System.Double.Nan if failed to communicate with the MP device
		/// </summary>
		public static double Temperature
		{
			get
			{
				MPCODE rval;
				double[] temp = new double[16];

				if((rval = MP.getMostRecentSample(temp)) == MP.MPRETURNCODE.MPSUCCESS)
					//scale MP data so that 0 maps to 90 and 1 maps to 95
					return temp[0] * 5 + 90;

				return System.Double.NaN;
			}

		}
		#endregion

		#region Public methods
		/// <summary>
		/// Initialize the acquisition by connecting to the MP Device, setting
		/// sample rate to 1 Hz and configuring it to acquire analog channel 1
		/// </summary>
		/// <param name="SN">the serial number of the MP160, if necessary</param>
		/// <returns>true, if the MP is configured properly</returns>
		public static bool Init(string SN)
		{
			MPCODE rval;
			bool[] achannel = new bool[16];

			//connect to mp160
			if((rval = MP.connectMPDev(MP.MPTYPE.MP160, MP.MPCOMTYPE.MPUDP, SN)) != MPCODE.MPSUCCESS)
			{
				StopMonitor();
				return false;
			}

			//set sample rate at 1 Hz
			if((rval = MP.setSampleRate(1000/1.0)) != MPCODE.MPSUCCESS)
			{
				StopMonitor();
				return false;
			}
			
			//acquire on channel 1 (by default all channels are false)
			achannel[0] = true;

			if((rval = MP.setAcqChannels(achannel)) != MPCODE.MPSUCCESS)
			{
				StopMonitor();
				return false;
			}

			//initialize LED segment array
			InitTensSegmentJaggedArray();
			InitOnesSegmentJaggedArray();

			return true;	
		}

		/// <summary>
		/// Starts the acquisition
		/// </summary>
		/// <returns>true, for success</returns>
		public static bool StartMonitor()
		{
			MPCODE rval;
			
			if((rval = MP.startAcquisition())!= MPCODE.MPSUCCESS)
			{
				StopMonitor();
				return false;
			}

			return true;
		}

		/// <summary>
		/// Stops the acquisition and disconnect from the MP device
		/// </summary>
		public static void StopMonitor()
		{
			OutputToA1(0.0);
			fan = false;
			MP.stopAcquisition();
			MP.disconnectMPDev();
		}

		/// <summary>
		/// Sends the current temperature to the LEDs
		/// </summary>
		/// <returns>true, for success</returns>
		public static bool SendToLED()
		{
			uint ival = (uint) TempControl.Temperature;
			uint tens = 0;
			uint ones = 0;

			//only send if the current Temperature is not the same value
			if(ledCurrVal == ival)
				return true;
			
			ledCurrVal = ival;
			
			tens = (ival / 10) % 10;
			ones = ival % 10;

		     ResetDigVal();
			//digital lines are all false
			//this will just turn on the necesseray led segments
			if(!SendToTensLED(tens) || !SendToOnesLED(ones))
			{
				StopMonitor();
				return false;
			}
				
			return true;
    	}

		/// <summary>
		/// Activates or Deactivates the Fan
		/// Sends 0 volts or 5.0 volts to analog out 1
		/// </summary>
		/// <param name="threshold">threshold greater than current temperature turn on fan else turn it off</param>
		/// <returns>true, for success</returns>
		public static bool ActivateFan(double threshold)
		{
			double thres = threshold;
			double temp =  TempControl.Temperature;

			//turn fan off case
			if(temp <= thres)
			{
				//if off keep it off
				if(!fan)
					return false;
				else
				{
					//turn it off
					if(!OutputToA1(0.0))
					{
						MP.stopAcquisition();
						MP.disconnectMPDev();
						return false;
					}

					fan = false;
				}
			}
			//turn fan on case
			else
			{
				//Only turn it on if it's not already on
				if(!fan)
				{
					if(!OutputToA1(5.0))
					{
						MP.stopAcquisition();
						MP.disconnectMPDev();
						return false;
					}

					fan = true;
				}
			}
			
			return true;
		}
		#endregion

		#region Private methods
		/// <summary>
		/// Sends specified voltage to Analog Channel 1
		/// </summary>
		/// <param name="volt">voltage level</param>
		/// <returns>tru,e for success </returns>
		private static bool OutputToA1(double volt)
		{
			MPCODE rval;

			if(-10.0 > volt || volt > 10.0)
				return false;

			if((rval = MP.setAnalogOut(volt,1)) != MPCODE.MPSUCCESS)
				return false;

			return true;
		}

		/// <summary>
		/// Turn on the LED segments that represents the digit in the tens place
		/// </summary>
		/// <param name="digit">digit to represent</param>
		/// <returns>true, for succes</returns>
		private static bool SendToTensLED(uint digit)
		{
			MPCODE rval;
			MPDIGOPT opt;

			foreach(uint d in tensSegments[digit])
			{
				opt = (d < 8) ? MPDIGOPT.SET_LOW_BITS : MPDIGOPT.SET_HIGH_BITS; 
				
				if((rval = MP.setDigitalIO(d,true,true, opt)) != MPCODE.MPSUCCESS)
					return false;
			}

			return true;
		}

		/// <summary>
		/// Turn on the LED segments that represents the digit in the ones place
		/// </summary>
		/// <param name="digit">digit to represent</param>
		/// <returns>true, for succes</returns>
		private static bool SendToOnesLED(uint digit)
		{
			MPCODE rval;
			MPDIGOPT opt;

			foreach(uint d in onesSegments[digit])
			{
				opt = (d < 8) ? MPDIGOPT.SET_LOW_BITS : MPDIGOPT.SET_HIGH_BITS; 
				
				if((rval = MP.setDigitalIO(d,true,true, opt)) != MPCODE.MPSUCCESS)
					return false;
			}
			
			return true;
		}

		/// <summary>
		/// Reset all digital lines to false
		/// This function does not send anything to the MP device
		/// </summary>
		private static void ResetDigVal()
		{
			MPCODE rval = MPCODE.MPSUCCESS;

			//clear LEDs reset a
			for(uint i=0; i < 16; i++)
				rval = MP.setDigitalIO(i, false,false,MPDIGOPT.SET_LOW_BITS);
			
		}

		/// <summary>
		/// Initialize the arrays that represents the digital line to turn
		/// on in order to represent a digit in the tens place LED
		/// </summary>
		private static void InitTensSegmentJaggedArray()
		{
			tensSegments = new uint[10][];

			tensSegments[0] = new uint[] {4,5,6,7,13,15};
			tensSegments[1] = new uint[] {7,13};
			tensSegments[2] = new uint[] {6,7,12,4,15};
			tensSegments[3] = new uint[] {6,7,12,13,15};
			tensSegments[4] = new uint[] {5,12,7,13};
			tensSegments[5] = new uint[] {6,5,12,13,15};
			tensSegments[6] = new uint[] {6,5,4,15,13,12};
			tensSegments[7] = new uint[] {6,7,13};
			tensSegments[8] = new uint[] {4,5,6,7,12,13,15};
			tensSegments[9] = new uint[] {6,5,12,7,13,15};
		} 

		/// <summary>
		/// Initialize the arrays that represents the digital line to turn
		/// on in order to represent a digit in the ones place LED
		/// </summary>
		private static void InitOnesSegmentJaggedArray()
		{
			onesSegments = new uint[10][];

			onesSegments[0] = new uint[] {11,10,9,8,2,0};
			onesSegments[1] = new uint[] {8,2};
			onesSegments[2] = new uint[] {9,8,3,11,0};
			onesSegments[3] = new uint[] {9,8,3,2,0};
			onesSegments[4] = new uint[] {10,3,8,2};
			onesSegments[5] = new uint[] {9,10,3,2,0};
			onesSegments[6] = new uint[] {9,10,11,0,2,3};
			onesSegments[7] = new uint[] {9,8,2};
			onesSegments[8] = new uint[] {11,10,9,8,3,2,0};
			onesSegments[9] = new uint[] {9,10,3,8,2,0};
		}
		#endregion
	}
}
