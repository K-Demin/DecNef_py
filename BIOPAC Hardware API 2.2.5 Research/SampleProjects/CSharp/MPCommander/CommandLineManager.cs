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
using System.IO;
using System.Windows.Forms;
using System.Reflection;

namespace MPCommander
{
	
	/// <summary>
	/// Description of CommandLineManager.
	/// </summary>
	public class CommandLineManager
	{
		private MPDevice MP;
		
		/// <summary>
		/// Command return codes
		/// </summary>
		private enum Command
		{
			Exit,		// Exit Command
			Error,		// System Error Need to Exit
			Success,	// Command is defined and had been executed
			Undefined	// Command is undefined
		}
	
		/// <summary>
		/// CommandLineManager constructor
		/// </summary>
		public CommandLineManager()
		{
			MP = new MPDevice();	//make sure that the MP Device is null until configuration
		
			Console.WriteLine("\nWelcome to MP Commander!!!");
			Console.WriteLine("--------------------------");
			PrintHelp();
		}
		
		/// <summary>
		/// Starts the CommandLineManager
		/// </summary>
		public void Start()
		{
			Command com = Command.Undefined;
			string input = "";
			
			ExtractResources();
			
			do
			{
				PrintCmdPrompt("");
				input = WaitForCmd();
				com = ProcessCommand(input);
			}
			while(com != Command.Exit);
			
			//clean up first before removing resources
			CleanUpMPDevice();
			
			//RemoveResources();
			
			return;
		}
		
		/// <summary>
		/// Writes the required resource files to directory
		/// where the executable resides
		/// </summary>
		private void ExtractResources()
		{
			string outdir = Application.StartupPath + @"\resource";
			
			Assembly thisAssembly =this.GetType().Assembly;
			string[] resources = thisAssembly.GetManifestResourceNames();
			
			Directory.CreateDirectory(outdir);

			if(resources != null)
			{
				for(int i=0; i<resources.Length; i++)
				{				
					string resourcepath = outdir + @"\" + resources[i];
																		
					FileStream fs = File.Create(resourcepath);
					StreamReader streamReader = new StreamReader(thisAssembly.GetManifestResourceStream(resources[i]));
					StreamWriter streamWriter = new StreamWriter(fs);

					streamWriter.Write(streamReader.ReadToEnd());
					
					streamReader.Close();
					streamWriter.Close();
					fs.Close();	
				}
			}
		}
		
		private void RemoveResources()
		{
			Directory.Delete(Application.StartupPath + @"\resource",true);
		}
		
		/// <summary>
		/// Process Commands
		/// </summary>
		/// <param name="input">command line input</param>
		/// <returns>the command status</returns>
		private Command ProcessCommand(string input)
		{
			Command com = Command.Undefined;
			string cleaninput = SanitizeInput(input);
			string[] args = cleaninput.Split(' ');
			
			switch(args[0].ToUpper())
			{
				//case "MP35USB":
				//	com = ExecuteMP35USB();
				//	break;
				case "MP160UDP":
					com = ExecuteMP150UDP(cleaninput);
					break;
                case "MP36USB":
                    com = ExecuteMP36USB();
                    break;
                case "GETANALOG":
					com = ExecuteGetAnalog(cleaninput);
					break;
				case "GETDIGITAL":
					com = ExecuteGetDigital(cleaninput);
					break;
				case "SETANALOG":
					com = ExecuteSetAnalog(cleaninput);
					break;
				case "SETDIGITAL":
					com = ExecuteSetDigital(cleaninput);
					break;
				case "SETPRESET":
					com = ExecuteSetPreset(cleaninput);
					break;
				case "HELP":
					PrintHelp();
					com = Command.Success;
					break;
				case "EXIT":
					com = Command.Exit;
					break;
				
				default:
					com = Command.Undefined;
					Console.WriteLine("'{0}' is an undefined command.", args[0]);
					break;
			}
			
			return com;
		}
		#region Execute Functions
		/// <summary>
		/// Excutes the SETPRESET command
		/// </summary>
		/// <param name="input">assumes a sanitized version of the commandline input</param>
		/// <returns>command status</returns>
		private Command ExecuteSetPreset(string input)
		{
			string[] args = input.Split(' ');
			
			if(!MP.IsConnected())
			{
				CleanUpMPDevice();
				return Command.Error;
			}
			
			if(args.Length < 3)
			{
				Console.WriteLine("Missing paramater(s).");
				PrintSetDigitalHelp();
				return Command.Error;
			}
			
			Command com = Command.Error;
			
			int ch = -1;
						
			try
			{
				ch = System.Int32.Parse(args[2]);
			}
			catch(System.FormatException)
			{
				com = Command.Error;
				Console.WriteLine("Invalid paramater {0}.",args[2]);
				PrintSetPresetHelp();
			}
			
			com = (MP.SetPreset(args[1],ch)) ? Command.Success : Command.Error;		
			
			return com;
		}
		
		/// <summary>
		/// Excutes the SETDIGITAL command
		/// </summary>
		/// <param name="input">assumes a sanitized version of the commandline input</param>
		/// <returns>command status</returns>
		private Command ExecuteSetDigital(string input)
		{
			string[] args = input.Split(' ');
			
			if(!MP.IsConnected())
			{
				CleanUpMPDevice();
				return Command.Error;
			}
			
			if(args.Length < 3)
			{
				Console.WriteLine("Missing paramater(s).");
				PrintSetDigitalHelp();
				return Command.Error;
			}
			
			Command com = Command.Error;
			
			switch(args[1].ToUpper())
			{
				case "ALL":
					{
						bool level;
						
						try
						{
							level = System.Boolean.Parse(args[2]);
						}
						catch(System.FormatException)
						{
							com = Command.Error;
							Console.WriteLine("Invalid paramater {0}.",args[2]);
							PrintSetDigitalHelp();
							break;
						}
						
						com = (MP.SetAllDigitalOutputChannels(level)) ? Command.Success : Command.Error;
						
						break;
					}
				default:
					{
						int ch;
						bool level;
						
						try
						{
							ch = System.Int32.Parse(args[1]);
						}
						catch(System.FormatException)
						{
							com = Command.Error;
							Console.WriteLine("Invalid paramater {0}.",args[1]);
							PrintSetDigitalHelp();
							break;
						}
						
						try
						{
							level = System.Boolean.Parse(args[2]);
						}
						catch(System.FormatException)
						{
							com = Command.Error;
							Console.WriteLine("Invalid paramater {0}.",args[2]);
							PrintSetDigitalHelp();
							break;
						}
						
						com = (MP.SetDigitalOutputChannel(ch,level)) ? Command.Success : Command.Error;
						
						break;
					}
			}
			
			return com;
		}
		
		/// <summary>
		/// Excutes the GETDIGITAL command
		/// </summary>
		/// <param name="input">assumes a sanitized version of the commandline input</param>
		/// <returns>command status</returns>
		private Command ExecuteGetDigital(string input)
		{
			string[] args = input.Split(' ');
			
			if(!MP.IsConnected())
			{
				CleanUpMPDevice();
				return Command.Error;
			}
			
			if(args.Length < 3)
			{
				Console.WriteLine("Missing paramater(s).");
				PrintGetDigitalHelp();
				return Command.Error;
			}
			
			Command com = Command.Error;
			
			bool online = false;
			
			if(args[1].ToUpper() == "ONLINE")
				online = true;
			else if(args[1].ToUpper() == "OFFLINE")
				online = false;
			else
			{
				Console.WriteLine("Invalid paramater {0}.",args[1]);
				PrintGetDigitalHelp();
				return Command.Error;
			}
			
			switch(args[2].ToUpper())
			{
				case "ALL":
					{
						com = (MP.PrintAllDigitalChannels(online)) ? Command.Success : Command.Error;
						break;
					}
				default:
					{
						int ch; 
						
						try
						{
							ch = System.Int32.Parse(args[2]);
						}
						catch(System.FormatException)
						{
							com = Command.Error;
							Console.WriteLine("Invalid paramater {0}.",args[2]);
							PrintGetDigitalHelp();
							break;
						}
						
						com = (MP.PrintDigitalChannel(online,ch)) ? Command.Success : Command.Error;
						break;
					}
			}
				
			return com;
		}
		
		/// <summary>
		/// Excutes the SETANALOG command
		/// </summary>
		/// <param name="input">assumes a sanitized version of the commandline input</param>
		/// <returns>command status</returns>
		private Command ExecuteSetAnalog(string input)
		{
			string[] args = input.Split(' ');
			
			if(!MP.IsConnected())
			{
				CleanUpMPDevice();
				return Command.Error;
			}
			
			if(args.Length < 3)
			{
				Console.WriteLine("Missing paramater(s).");
				PrintSetAnalogHelp();
				return Command.Error;
			}
			
			Command com = Command.Error;
			
			switch(args[1].ToUpper())
			{
				case "ALL":
					{
						double voltage;
						
						try
						{
							voltage = System.Double.Parse(args[2]);
						}
						catch(System.FormatException)
						{
							com = Command.Error;
							Console.WriteLine("Invalid paramater {0}.",args[2]);
							PrintSetAnalogHelp();
							break;
						}
						
						com = (MP.SetAllAnalogOutputChannels(voltage)) ? Command.Success : Command.Error;
						
						break;
					}
				default:
					{
						int ch;
						double voltage;
						
						try
						{
							ch = System.Int32.Parse(args[1]);
						}
						catch(System.FormatException)
						{
							com = Command.Error;
							Console.WriteLine("Invalid paramater {0}.",args[1]);
							PrintSetAnalogHelp();
							break;
						}
						
						try
						{
							voltage = System.Double.Parse(args[2]);
						}
						catch(System.FormatException)
						{
							com = Command.Error;
							Console.WriteLine("Invalid paramater {0}.",args[2]);
							PrintSetAnalogHelp();
							break;
						}
						
						com = (MP.SetAnalogOutputChannel(ch,voltage)) ? Command.Success : Command.Error;
						
						break;
					}
			}
			
			return com;
		}
		
		/// <summary>
		/// Excutes the GETANALOG command
		/// </summary>
		/// <param name="input">assumes a sanitized version of the commandline input</param>
		/// <returns>command status</returns>
		private Command ExecuteGetAnalog(string input)
		{
			string[] args = input.Split(' ');
			
			if(!MP.IsConnected())
			{
				CleanUpMPDevice();
				return Command.Error;
			}
			
			if(args.Length < 2)
			{
				Console.WriteLine("Missing paramater.");
				PrintGetAnalogHelp();
				return Command.Error;
			}
			
			Command com = Command.Error;
			
			switch(args[1].ToUpper())
			{
				case "ALL":
					{
						com = (MP.PrintAllAnalogChannels()) ? Command.Success : Command.Error;
						break;
					}
				default:
					{
						int ch; 
						
						try
						{
							ch = System.Int32.Parse(args[1]);
						}
						catch(System.FormatException)
						{
							com = Command.Error;
							Console.WriteLine("Invalid paramater {0}.",args[1]);
							PrintGetAnalogHelp();
							break;
						}
						
						com = (MP.PrintAnalogChannel(ch)) ? Command.Success : Command.Error;
						break;
					}
			}
				
			return com;
		}
		
		/// <summary>
		/// Excutes the MP35USB command
		/// </summary>
		/// <returns>command status</returns>
	/*
		private Command ExecuteMP35USB()
		{
			//disconnect from the previous MP Device
			MP.Disconnect();
			
			//second arguement should be serial number
			MP = new MP35USB();
			
			if(!MP.Connect(""))
			{
				PrintMP35USB();
				CleanUpMPDevice();
				return Command.Error;
			}
			
			if(!MP.Configure())
			{
				CleanUpMPDevice();
				return Command.Error;
			}
			
			Console.WriteLine("Connected to the MP35");
			Console.WriteLine("MP35 has been successfully configured.");
			
			return Command.Success;
		}
	*/

        /// <summary>
        /// Excutes the MP36USB command
        /// </summary>
        /// <returns>command status</returns>
        private Command ExecuteMP36USB()
        {
            //disconnect from the previous MP Device
            MP.Disconnect();

            //second arguement should be serial number
            MP = new MP36USB();

            if (!MP.Connect(""))
            {
                PrintMP36USB();
                CleanUpMPDevice();
                return Command.Error;
            }

            if (!MP.Configure())
            {
                CleanUpMPDevice();
                return Command.Error;
            }

            Console.WriteLine("Connected to the MP36");
            Console.WriteLine("MP36 has been successfully configured.");

            return Command.Success;
        }

        /// <summary>
		/// Excutes the MP150UDP command
		/// </summary>
		/// <param name="input">assumes a sanitized version of the commandline input</param>
		/// <returns>command status</returns>
		private Command ExecuteMP150UDP(string input)
		{
			//disconnect from the previous MP Device
			MP.Disconnect();
			
			string[] args = input.Split(' ');
			
			if(args.Length < 2)
			{
				Console.WriteLine("MP160 serial number required.");
				PrintMP150UDPHelp();
				return Command.Error;
			}
			
			//second arguement should be serial number
			MP = new MP150UDP();
			
			if(!MP.Connect(args[1]))
			{
				PrintMP150UDPHelp();
				CleanUpMPDevice();
				return Command.Error;
			}
			
			if(!MP.Configure())
			{
				CleanUpMPDevice();
				return Command.Error;
			}
			
			Console.WriteLine("Connected to the MP160 with serial number '{0}'", args[1]);
			Console.WriteLine("MP160 has been successfully configured.");
			
			return Command.Success;
		}
		#endregion
		#region Aux Functions
		/// <summary>
		/// Removes trailing and leading spaces and replace multiple space between words with one space
		/// </summary>
		/// <param name="input">a string</param>
		/// <returns></returns>
		private string SanitizeInput(string input)
		{
			//remove leading and trailing whitespace
			string sanitized = input.Trim();
			
			//replace multiple whitespace inside the input with single white space 
			sanitized = System.Text.RegularExpressions.Regex .Replace(sanitized, @"[\s]+", " ");
			
			return sanitized;
		}
		private void CleanUpMPDevice()
		{
			MP.Disconnect();
			MP = new MPDevice();
		}
		#endregion
		#region Print Functions
		private void PrintSetPresetHelp()
		{
			Console.WriteLine("Correct syntax: setpreset presetid channelnumber");
			Console.WriteLine("presetid\t\tThe unique preset id.  See 'channelpresets.xml'");
			Console.WriteLine("channelnumber\tthe numeric value of the analog output channel starting from 0 to be set");
			Console.WriteLine("NOTE: For the MP160, only the scaling is applied.");
			Console.WriteLine("Example:");
			Console.WriteLine("setpreset a15 0");
		}
		private void PrintSetDigitalHelp()
		{
			Console.WriteLine("Correct syntax: setdigital all|channelnumber level");
			Console.WriteLine("all\t\tsets the values of all the digital output channels");
			Console.WriteLine("channelnumber\tthe numeric value of the digital output channel starting from 0 to be set");
			Console.WriteLine("level\t\tthe level of the digital output channel to be set TRUE->HIGH FALSE->LOW");
			Console.WriteLine("Example:" );
			Console.WriteLine("setdigital all false");
			Console.WriteLine("setdigital 0 true");
		}
		private void PrintGetDigitalHelp()
		{
			Console.WriteLine("Correct syntax: getdigital online|offline all|channelnumber");
			Console.WriteLine("online\t\tget the digital channel(s) by starting acquisition and reading the values of the digital channels as samples");
			Console.WriteLine("offline\t\tget the digtal channel(s) without starting an acquisition.");
			Console.WriteLine("all\t\tprints the values of all the analog input channels");
			Console.WriteLine("channelnumber\tthe numeric value of the digital input channel starting from 0 to be read");
			Console.WriteLine("Example:" );
			Console.WriteLine("getdigital online all");
			Console.WriteLine("getdigital offline 0");
			Console.WriteLine("getdigital online 2");
		}
		private void PrintSetAnalogHelp()
		{
			Console.WriteLine("Correct syntax: setanalog all|channelnumber voltage");
			Console.WriteLine("all\t\tsets the values of all the analog output channels");
			Console.WriteLine("channelnumber\tthe numeric value of the analog output channel starting from 0 to be set");
			Console.WriteLine("voltage\t\tthe voltage value of the analog output channel to be set in volts");
			Console.WriteLine("Example:" );
			Console.WriteLine("setanalog all");
			Console.WriteLine("setanalog 0 1.0");
		}
		private void PrintGetAnalogHelp()
		{
			Console.WriteLine("Correct syntax: getanalog all|channelnumber");
			Console.WriteLine("all\t\tprints the values of all the analog input channels");
			Console.WriteLine("channelnumber\tthe numeric value of the analog input channel starting from 0 to be read");
			Console.WriteLine("Example:" );
			Console.WriteLine("getanalog all");
			Console.WriteLine("getanalog 0");
		}
		
		private void PrintMP35USB()
		{
			Console.WriteLine("Correct syntax: mp35usb");
			Console.WriteLine("Example:" );
			Console.WriteLine("mp35usb");
		}

        private void PrintMP36USB()
        {
            Console.WriteLine("Correct syntax: mp36usb");
            Console.WriteLine("Example:");
            Console.WriteLine("mp36usb");
        }

        private void PrintMP150UDPHelp()
		{
			Console.WriteLine("Correct syntax: mp160udp SERIALNUMBER");
			Console.WriteLine("SERIALNUMBER\t the serial number of the MP160.");
			Console.WriteLine("Example:" );
			Console.WriteLine("mp160udp 308A-0000393");
		}
		
		private void PrintHelp()
		{
			Console.WriteLine("Command Set:");
			Console.WriteLine("============");
			Console.WriteLine("exit\t\tExit and quit MP Commander.");
			Console.WriteLine("getanalog\tFetches the values of the MP Device's analog input channel(s).");
			Console.WriteLine("getdigital\tFetches the values of the MP Device's digital input channel(s).");
			Console.WriteLine("help\t\tPrints help information.");
			Console.WriteLine("mp160udp\tConnect to the MP160 via Ethernet using UDP.");
//			Console.WriteLine("mp35usb\t\tConnect to the MP35 via USB.");
            Console.WriteLine("mp36usb\t\tConnect to the MP36 via USB.");
            Console.WriteLine("setanalog\tSets the values of the MP Device's analog output channel(s).");
			Console.WriteLine("setdigital\tSets the values of the MP Device's digital output channel(s).");
			Console.WriteLine("setpreset\tSets the preset for the specified MP Device analog input channel.");
		}
		
		private void PrintCmdPrompt(string prompt)
		{
			Console.Write("{0}>",prompt);
		}
		
		private string WaitForCmd()
		{ 
			return Console.ReadLine();
		}
		#endregion Region
	}
}
