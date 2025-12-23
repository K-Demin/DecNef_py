''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Copyright 2005-2023 BIOPAC Systems, Inc.
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
' Portions Copyright 2005-2023 BIOPAC Systems, Inc.
'
' 2. Altered source versions must be plainly marked as such, and must not be 
' misrepresented as being the original software.
'
' 3. This notice may not be removed or altered from any source distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Module BHAPIBasics
    Private mpconfig As String 'string representation on the mp configuration
    Private numAnalogCh As Integer 'number of analog channels
    Private numDigitalCh As Integer 'number of digital channels
    Private sampleRate As Double 'in Hz
    Private aCH(15) As Int32 'represents the analog acquisition channels
    Private dCH(15) As Int32 'represents the digital acquisition channels

    Sub Main()
        mpconfig = ""

        'force the user to enter a valid response to the prompt
        While Not PormptForMPConfig()
        End While

        'process the prompt if it is exit
        If mpconfig.Equals("EXIT") Then
            PrintMessage("Exiting...")
            Return
        End If

        PrintMessage("Connecting to " + mpconfig + "...")

        'connect to mp device
        If Not ConnectToMPDevice() Then
            PrintMessage("Failed to connect.")
            Return
        End If

        PrintMessage("Connected...")

        PrintMessage("Configuring acquisition...")

        'configure the acquisition
        If Not (ConfigureAcquisition()) Then
            PrintMessage("Failed to configure.")
            Return
        End If

        PrintMessage("Acquisition configured...")

        'execute get most recent sample demo function
        If Not AcquireUsingGetMostRecentSample() Then
            DisconnectFromMPDevice()
            Return
        End If

        'execute get mp buffer demo function
        If Not AcquireUsingGetMPBuffer() Then
            DisconnectFromMPDevice()
            Return
        End If

        'cleanly disconnect from the API
        DisconnectFromMPDevice()

        PrintMessage("Disconnected...")
    End Sub

    Private Function AcquireUsingGetMPBuffer()
        PrintMessage("--Acquire Data Using Get Most Recent Sample---")

        PrintMessage("Start Acquisition...")

        If Not (StartAcquisition()) Then
            PrintMessage("Failed to start acquisition.")
            Return False
        End If

        Dim retval As MPRETURNCODE
        Dim i As Integer
        Dim numActiveAnalog As Integer = 0
        Dim numActiveDigital As Integer = 0

        'count the number of active analog and digital channels
        For i = 0 To aCH.Length - 1
            If aCH(i) = 1 Then
                numActiveAnalog = numActiveAnalog + 1
            End If
        Next

        For i = 0 To dCH.Length - 1
            If dCH(i) = 1 Then
                numActiveDigital = numActiveDigital + 1
            End If
        Next

        'allocate 2 seconds of data
        Dim numberOfSeconds As Double = 2.0
        Dim buff((sampleRate * (numActiveAnalog + numActiveDigital) * numberOfSeconds) - 1) As Double

        'acquire 2 second of data
        Dim numberOfSamples As Integer = sampleRate * numberOfSeconds
        retval = MPDev.getMPBuffer(numberOfSamples, buff(0))

        If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
            PrintMessage("Failed to get most recent sample.")
            StopAcquisition()
            Return False
        End If

        Dim sampleoffset As Integer
        Dim offset As Integer
        Dim c As Integer
        'print first 10 sample
        For i = 0 To 9
            PrintMessage("Sample " + (i + 1).ToString())
            sampleoffset = (numActiveAnalog + numActiveDigital) * i

            offset = sampleoffset
            For c = 0 To aCH.Length - 1
                If aCH(c) = 1 Then
                    PrintMessage("Analog CH" + (c + 1).ToString() + "= " + buff(offset).ToString())
                    offset = offset + 1
                End If
            Next

            offset = numActiveDigital + sampleoffset
            For c = 0 To dCH.Length - 1
                If dCH(c) = 1 Then
                    PrintMessage("Digital CH" + (c).ToString() + "= " + buff(offset).ToString())
                    offset = offset + 1
                End If
            Next

        Next

        StopAcquisition()

        PrintMessage("Acquisition stopped.")

        Return True
    End Function

    Private Function AcquireUsingGetMostRecentSample() As Boolean
        PrintMessage("--Acquire Data Using Get Most Recent Sample---")

        PrintMessage("Start Acquisition...")

        If Not (StartAcquisition()) Then
            PrintMessage("Failed to start acquisition.")
            Return False
        End If

        Dim retval As MPRETURNCODE
        Dim i As Integer
        Dim buff(numAnalogCh + numDigitalCh - 1) As Double

        For i = 0 To 9
            retval = MPDev.getMostRecentSample(buff(0))

            If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
                PrintMessage("Failed to get most recent sample.")
                StopAcquisition()
                Return False
            End If

            Dim c As Int32

            For c = 0 To aCH.Length - 1
                If aCH(c) = 1 Then
                    PrintMessage("Analog CH" + (c + 1).ToString() + "= " + buff(c).ToString())
                End If
            Next

            For c = aCH.Length To buff.Length - 1
                If dCH(c - aCH.Length) = 1 Then
                    PrintMessage("Digital CH" + (c - aCH.Length).ToString() + "= " + buff(c).ToString())
                End If
            Next
        Next

        StopAcquisition()

        PrintMessage("Acquisition stopped.")

        Return True
    End Function

    Private Function ConfigureAcquisition() As Boolean
        ReDim aCH(numAnalogCh - 1)
        ReDim dCH(numDigitalCh - 1)
        Dim c As Int32
        Dim retval As MPRETURNCODE

        ' 0 implies false
        ' 1 implies true

        'initialize analog channels to false
        For c = 0 To aCH.Length - 1
            aCH(c) = 0
        Next

        'initialize digital channels to false
        For c = 0 To dCH.Length - 1
            dCH(c) = 0
        Next

        'set analog channel
        aCH(0) = 1 'acquire on analog channel 1
        aCH(1) = 1 'acquire on analog channel 2
        retval = MPDev.setAcqChannels(aCH(0))

        If Not (retval = MPRETURNCODE.MPSUCCESS) Then
            Return False
        End If

        dCH(1) = 1 'acquire on digital channel 1
        dCH(3) = 1 'acquire on digital channel 3

        retval = MPDev.setDigitalAcqChannels(dCH(0))

        If Not (retval = MPRETURNCODE.MPSUCCESS) Then
            Return False
        End If

        'set sample rate
        sampleRate = 1000.0 'Hz

        retval = MPDev.setSampleRate(1000.0 / sampleRate)

        If Not (retval = MPRETURNCODE.MPSUCCESS) Then
            Return False
        End If

        Return True
    End Function

    Private Sub StopAcquisition()
        MPDev.stopAcquisition()
    End Sub

    Private Function StartAcquisition() As Boolean
        Return (MPDev.startAcquisition() = MPDev.MPRETURNCODE.MPSUCCESS)
    End Function


    Private Function ConnectToMPDevice() As Boolean
        Dim retval As MPRETURNCODE

        If mpconfig.Equals("MP160UDP") Then
            'if this function fails replace "auto" with the full serial number of your MP160
            retval = MPDev.connectMPDev(MPTYPE.MP160, MPCOMTYPE.MPUDP, "auto")
        End If

        If mpconfig.Equals("MP200UDP") Then
            'if this function fails replace "auto" with the full serial number of your MP160
            retval = MPDev.connectMPDev(MPTYPE.MP200, MPCOMTYPE.MPUDP, "auto")
        End If

        ' If mpconfig.Equals("MP35USB") Then
        '     retval = MPDev.connectMPDev(MPTYPE.MP35, MPCOMTYPE.MPUSB, "auto")
        ' End If

        If mpconfig.Equals("MP36USB") Then
            retval = MPDev.connectMPDev(MPTYPE.MP36, MPCOMTYPE.MPUSB, "auto")
        End If

        If mpconfig.Equals("MP36A") Then
            retval = MPDev.connectMPDev(MPTYPE.MP36A, MPCOMTYPE.MPUSB, "auto")
        End If

        Return (retval = MPRETURNCODE.MPSUCCESS)

    End Function

    Private Sub DisconnectFromMPDevice()
        MPDev.disconnectMPDev()
    End Sub

    Private Function PormptForMPConfig() As Boolean
        Dim usrinput As String = ""
        mpconfig = ""

        System.Console.Write("Enter MP200UDP, MP160UDP, MP36USB, MP36A or EXIT: ")

        usrinput = System.Console.ReadLine().Trim()

        If usrinput.ToUpper.Equals("MP160UDP") Then
            mpconfig = "MP160UDP"
            numAnalogCh = 16
            numDigitalCh = 16
            Return True
        End If

        If usrinput.ToUpper.Equals("MP200UDP") Then
            mpconfig = "MP200UDP"
            numAnalogCh = 16
            numDigitalCh = 16
            Return True
        End If

        'If usrinput.ToUpper.Equals("MP35USB") Then
        '    mpconfig = "MP35USB"
        '    numAnalogCh = 4
        '    numDigitalCh = 8
        '    Return True
        ' End If

        If usrinput.ToUpper.Equals("MP36USB") Then
            mpconfig = "MP36USB"
            numAnalogCh = 4
            numDigitalCh = 8
            Return True
        End If

        If usrinput.ToUpper.Equals("MP36A") Then
            mpconfig = "MP36A"
            numAnalogCh = 4
            numDigitalCh = 8
            Return True
        End If

        If usrinput.ToUpper.Equals("EXIT") Then
            mpconfig = "EXIT"
            Return True
        End If

        Return False
    End Function

    Private Sub PrintMessage(ByVal message As String)
        System.Console.WriteLine(message)
    End Sub

End Module
