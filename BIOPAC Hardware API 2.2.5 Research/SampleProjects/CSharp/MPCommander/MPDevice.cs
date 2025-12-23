/*
Copyright 2005-2009 BIOPAC Systems, Inc.

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

Portions Copyright 2005-2009 BIOPAC Systems, Inc.

2. Altered source versions must be plainly marked as such, and must not be 
misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
*/

using System;

using MPDEV = Biopac.API.MPDevice.MPDevImports;
using MPCODE = Biopac.API.MPDevice.MPDevImports.MPRETURNCODE;

namespace MPCommander
{
	/// <summary>
	/// Description of MPDevice.
	/// </summary>
	public class MPDevice
	{
		public MPDevice()
		{
		}
		
		public virtual bool Connect(string SN)
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool Configure()
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool IsConnected()
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool PrintAllAnalogChannels()
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool PrintAnalogChannel(int ch)
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool SetAllAnalogOutputChannels(double voltage)
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool SetAnalogOutputChannel(int ch, double voltage)
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool PrintAllDigitalChannels(bool online)
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool PrintDigitalChannel(bool online, int ch)
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool SetAllDigitalOutputChannels(bool level)
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool SetDigitalOutputChannel(int ch, bool level)
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual bool SetPreset(string uid, int ch)
		{
			Console.WriteLine("Not connected to an MP Device.");
			
			return false;
		}
		
		public virtual void Disconnect()
		{
			//intentionally left empty
		}
	}
}
