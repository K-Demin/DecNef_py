BIOPAC HARDWARE API 2.2.5 for Windows
-----------------------

WELCOME
-------

This is the BIOPAC Hardware API.  The product will allow you to create custom software programs that can communicate directly with BIOPAC MP devices.

License
-------
- Please see "license.txt"
- In compliance with the Apache Xerces distribution license, "NOTICE" and "LICENSE" text files are included.

Documentation
-------------
Please open the file "BHAPIhelp.html" located in the same folder as this file (should be C:\Program Files\BIOPAC Systems, Inc\BIOPAC Hardware API [version] Research).  A shortcut to this file will be placed on the desktop when the software is installed.

Sample Projects
---------------
The BIOPAC Hardware API includes several Sample Projects written in various programming languages. Please see BHAPIhelp.html to learn more about the Sample Projects.

  The sample projects are located at: [INSTALL DIRECTORY]\SampleProjects

  FOLDER DESCRIPTIONS
  -------------------
  Documentation		Contains the BIOPAC Hardware API Reference Manual
  LanguageBindings	Contains Language Bindings (wrappers) for BHAPI in different programming languages
  PresetFiles		Contains files necessary for using the Channel Presets XML file
  SampleProjects	Contains the sample projects and its documentation
  VC10			Contains the libraries compiled with Microsoft Visual Studio (MSVS) 2010
  VC14			Contains the libraries compiled with MSVS 2019


Updates since API 2.2.4
---------------------

- MP200 support is added
- MP36A support is added
- Binaries made with MSVS 2010 (VC10) and MSVS 2019 (VC14)
- Some sample applications modified to work with MP200
- Updated documentation
- Sample projects rebuilt with MSVS 2019
- Tested with MATLAB R2024a
- Tested with LabVIEW 2023 Q3
- New Python sample project

Updates since API 2.2.3
---------------------

- Sample applications modified to work with MP160
- Fixed issues preventing MATLAB from loading libraries for MATLAB samples
- Updated some documentation
- Verified some samples in newer programming environments


Updates since API 2.2.2
---------------------

- Support for MP36R electrode checker added


Updates since API 2.1.1
---------------------

- Support for MP160 hardware added


Updates since API 2.1
---------------------
- Visual Studio 2010 support
- 64-bit support
- Bug fixes


Updates since API 2.0
---------------------
- Support for MP36R hardware added
- No support for MP35 hardware
- Executable files for 8 sample applications added to SampleProjects:
   - Cplusplus: mp1XXdemo
   - CSharp: Biofeedback
   - CSharp: GoalKick
   - CSharp: TemperatureControl
   - CSharp: VideoStimulusMP36
   - VBNET: bhapibasics
   - VBNET: FunctionGenerator
   - VBNET: ImageStimMP36


Updates since API 1.0
---------------------
- Support for MP36 hardware 
- Bug fixing for MP150 units on computers with multiple network adapters 
- Installer creates 2 shortcuts to Sample/Help browser: 
      On the desktop: label is "BHAPI 2.0 manual" 
      START program menu: the path is "Start\Programs\BIOPAC Hardware API 2.0\BHAPI 2.0 manual" 
- MP36 USB drivers are included into the BHAPI 2.0 for Windows Installer 
- New sample applications designed to work with MP36 hardware 
- CH to Output redirection added for MP36 hardware.
- New API call of setAnalogOutputMode()allows user to switch between 3 ouptut modes supported by MP36 hardware: 
      a) Constant level voltage output for MP36 hardware (OUTPUTVOLTAGELEVEL)
      b) Redirecting input channel signal to output channel 0 for MP36 hardware
      c) Ground all output signal to zero
- New sample applications (dedicated to work with MP36 device) 
      - C# project - VideoStimulusMP36 
      - VB.NET project - ImageStimMP36 
      - LabView project - getBufferDemoMP36 
      - LabView project - startAcqDaemonDemoMP36 
      - LabView project - temperatureDemo forMP36 

Known Issues
------------
- NONE
