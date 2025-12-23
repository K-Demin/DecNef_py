''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Copyright 2005-2024 BIOPAC Systems, Inc.
'
' This software is provided 'as-is', without any express or implied warranty.
' In no event will BIOPAC Systems, Inc. or BIOPAC Systems, Inc. employees be 
' held liable for any damages arising from the use of this software.
'
' Permission is granted to anyone to use this software for any purpose, 
' including commercial applications, and to alter it and redistribute it 
' freely, subject to the following restrictions:
'
' 1. The origin of this software must not be misrepresented; you must not 
' claim that you wrote the original software. If you use this software in a 
' product, an acknowledgment (see the following) in the product documentation
' is required.
'
' Portions Copyright 2005-2010 BIOPAC Systems, Inc.
'
' 2. Altered source versions must be plainly marked as such, and must not be 
' misrepresented as being the original software.
'
' 3. This notice may not be removed or altered from any source distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Imports System.Runtime.InteropServices

' Visual Basic .NET translation of mpdev.h.  A language binding for mpdev.dll is created when compiled
' Supports BHAPI 2.1 for Windows
' See BHAPI 2.1 for Windows documentation for full documentation
Module MPDev
    Const dllpath As String = "mpdev.dll"

    Public Enum MPTYPE
        MP150 = 101
        'MP35
        MP36
        MP160
        MP200
        MP36A
    End Enum


    Public Enum MPCOMTYPE
        MPUSB = 10
        MPUDP
    End Enum

    Public Enum TRIGGEROPT
        MPTRIGOFF = 1
        MPTRIGEXT
        MPTRIGACH
        MPTRIGDCH
    End Enum

    Public Enum DIGITALOPT
        SET_LOW_BITS = 1
        SET_HIGH_BITS
        READ_LOW_BITS
        READ_HIGH_BITS
    End Enum

    Public Enum MPRETURNCODE
	      MPSUCCESS = 1
		MPDRVERR
		MPDLLBUSY
		MPINVPARA
		MPNOTCON
		MPREADY
		MPWPRETRIG
		MPWTRIG
		MPBUSY
		MPNOACTCH
		MPCOMERR
		MPINVTYPE
		MPNOTINNET
		MPSMPLDLERR
		MPMEMALLOCERR
		MPSOCKERR
		MPUNDRFLOW
		MPPRESETERR
		MPPARSERERR 
    End Enum

    Public Enum MP3XOUTMODE
	OUTPUTVOLTAGELEVEL		= 2
	OUTPUTCHANNEL3			= 3
	OUTPUTCHANNEL1			= 5
	OUTPUTCHANNEL2			= 6
	OUTPUTCHANNEL4			= 7
	OUTPUTGROUND			= &H7f
    End Enum


    <DllImport(dllpath)> _
    Public Function configChannelByPresetID(ByVal n As Integer, ByVal uid As String) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function connectMPDev(ByVal type As MPTYPE, ByVal method As MPCOMTYPE, ByVal SN As String) As MPRETURNCODE
    End Function

    <DllImport(dllpath)> _
    Public Function disconnectMPDev() As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function findAllMP150() As MPRETURNCODE
    End Function

    <DllImport(dllpath)> _
    Public Function findAllMP160() As MPRETURNCODE
    End Function

    <DllImport(dllpath)> _
    Public Function findAllMP200() As MPRETURNCODE
    End Function
    
    <DllImport(dllpath)> _
    Public Function getChScaledInputRange(ByVal n As Integer, ByRef minRange As Double, ByRef maxRange As Double) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function getDigitalIO(ByVal n As Integer, ByRef state As Boolean, ByVal opt As DIGITALOPT) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function getMPBuffer(ByVal numSamples As Integer, ByRef buff As Double) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function getMPDaemonLastError() As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function getMostRecentSample(ByRef data As Double) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function getStatusMPDev() As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function loadXMLPresetFile(ByVal path As String) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function readAvailableMP150SN(ByRef data As Char, ByVal numbytesToRead As Integer, ByRef numbytesRead As Integer) As MPRETURNCODE
    End Function

    <DllImport(dllpath)> _
    Public Function readAvailableMP160SN(ByRef data As Char, ByVal numbytesToRead As Integer, ByRef numbytesRead As Integer) As MPRETURNCODE
    End Function

    <DllImport(dllpath)> _
    Public Function readAvailableMP200SN(ByRef data As Char, ByVal numbytesToRead As Integer, ByRef numbytesRead As Integer) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function receiveMPData(ByRef data As Double, ByVal numdatapoints As Integer, ByRef numreceived As Integer) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function setAcqChannels(ByRef analogCH As Int32) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function setAnalogChScale(ByVal unscaled1 As Int32, ByVal scaled1 As Int32, ByVal unscaled2 As Int32, ByVal scaled2 As Int32) As MPRETURNCODE
    End Function

    <DllImport(dllpath)> _
    Public Function setAnalogOut(ByVal value As Double, ByVal outchan As Integer) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function setAnalogOutputMode (ByVal mode as MP3XOUTMODE ) As MPRETURNCODE
    End Function

    <DllImport(dllpath)> _
    Public Function setDigitalAcqChannels(ByRef digitalCH As Int32) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function setDigitalIO(ByVal n As Integer, ByVal state As Boolean, ByVal setnow As Boolean, ByVal opt As DIGITALOPT) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function setMPTrigger(ByVal opt As TRIGGEROPT, ByVal posEdge As Boolean, ByVal level As Double, ByVal chNum As Integer) As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function setSampleRate(ByVal rate As Double) As MPRETURNCODE
    End Function

    <DllImport(dllpath)> _
    Public Function startAcquisition() As MPRETURNCODE
    End Function
	
    <DllImport(dllpath)> _
    Public Function startMPAcqDaemon() As MPRETURNCODE
    End Function

    <DllImport(dllpath)> _
    Public Function stopAcquisition() As MPRETURNCODE
    End Function
End Module
