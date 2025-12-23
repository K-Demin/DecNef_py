''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Copyright 2005-2008 BIOPAC Systems, Inc.
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
' Portions Copyright 2005-2008 BIOPAC Systems, Inc.
'
' 2. Altered source versions must be plainly marked as such, and must not be 
' misrepresented as being the original software.
'
' 3. This notice may not be removed or altered from any source distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Imports System.Drawing


Namespace Biopac.API.MPDevice

    'structure for holding acquisition settings
    Public Structure AcquisitionConfiguration
        Public aCH() As Int32 'analog channel configuration
        Public dCH() As Int32 'digital channel configuration
        Public numCH As Integer 'number of active channels (digital and analog)
        Public length As Integer 'acquisition length in seconds
        Public frequency As Double 'frequency in Hz 
    End Structure

    Public Class StimuliDisplay
        Inherits System.Windows.Forms.Form

        Private StimReporter As StimulusReport 'for the stimulus report
        Private AcqConfig As AcquisitionConfiguration 'acquisition configuration
        Private OutputDir As String 'directory where the output files will be stored
        Private QUIT As Boolean 'quit slideshow flag

#Region " Windows Form Designer generated code "

        Public Sub New(ByVal dir As String)
            MyBase.New()

            'This call is required by the Windows Form Designer.
            InitializeComponent()

            'Add any initialization after the InitializeComponent() call

            'put the maximized window in the upper left corner
            Left = 0
            Top = 0
            
            OutputDir = dir 'store directory
            QUIT = False 'init variable

            'Initialize Acquisition Configuration
            Dim c As Int32

            ReDim AcqConfig.aCH(3) '4 analog channels
            ReDim AcqConfig.dCH(7) '8 digital channels

            AcqConfig.aCH(0) = 1 'acquire on analog channel 1
            AcqConfig.aCH(1) = 0
            AcqConfig.aCH(2) = 0
            AcqConfig.aCH(3) = 0

            'acquire on all digital channels
            For c = 0 To 7
                AcqConfig.dCH(c) = 1
            Next

            AcqConfig.numCH = 9 '9 channels total. 8 digital and 1 analog
            AcqConfig.length = 5   'seconds 
            AcqConfig.frequency = 1000 'Hz

            Cursor.Hide()
        End Sub

        'Form overrides dispose to clean up the component list.
        Protected Overloads Overrides Sub Dispose(ByVal disposing As Boolean)
            'disconnect prior to exit
            MPDev.disconnectMPDev()
            Cursor.Show()

            If disposing Then
                If Not (components Is Nothing) Then
                    components.Dispose()
                End If
            End If
            MyBase.Dispose(disposing)
        End Sub

        'Required by the Windows Form Designer
        Private components As System.ComponentModel.IContainer

        'NOTE: The following procedure is required by the Windows Form Designer
        'It can be modified using the Windows Form Designer.  
        'Do not modify it using the code editor.
        <System.Diagnostics.DebuggerStepThrough()> Private Sub InitializeComponent()
            '
            'StimuliDisplay
            '
            Me.AutoScaleBaseSize = New System.Drawing.Size(5, 13)
            Me.BackColor = System.Drawing.Color.Black
            Me.ClientSize = New System.Drawing.Size(292, 273)
            Me.FormBorderStyle = System.Windows.Forms.FormBorderStyle.None
            Me.Name = "StimuliDisplay"
            Me.Text = "Stimulus Display"
            Me.WindowState = System.Windows.Forms.FormWindowState.Maximized

        End Sub

#End Region

        Public Function SetupStimDisplay(ByVal imageList As ArrayList) As Boolean
            'preload the images and configure the acquisition
            Return PreLoadImageStim(imageList) And ConfigureAcquisition()
        End Function
        Public Sub Start()
            'calculate the buffer size
            Dim bufferSize As Integer = AcqConfig.numCH * AcqConfig.frequency * AcqConfig.length
            Dim count As Integer = StimReporter.StimulusList.Length
            Dim c As Integer
            Dim buffer(bufferSize - 1) As Double
            Dim retval As MPDev.MPRETURNCODE

            'for all the images in the list
            For c = 0 To count - 1
                'if quit flag is false
                If Not (QUIT) Then
                    ClearScreen()
                    'show the image
                    DisplayImage(c)
                    'start the acquisition
                    retval = MPDev.startAcquisition()

                    If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
                        Return
                    End If

                    'get the first five second buffer
                    retval = MPDev.getMPBuffer(AcqConfig.frequency * AcqConfig.length, buffer(0))

                    If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
                        Return
                    End If

                    'clear the screen when as soon as acquisition ends
                    ClearScreen()
                    'stop the acquisition
                    MPDev.stopAcquisition()
                    'get the physiological response
                    GetAnalogChannelZero(StimReporter.StimulusList(c).Waveform, buffer, bufferSize - 1)
                    'get the subjective response
                    StimReporter.StimulusList(c).Score = GetScore(buffer, bufferSize - 1)
                    Application.DoEvents()
                Else
                    Return
                End If
            Next

            'genearte the report
            StimReporter.GenerateReport()

            'disconnect from the BHAPI
            MPDev.disconnectMPDev()
        End Sub
        Public Sub Abort()
            'stop the acquisition
            MPDev.stopAcquisition()
            'disconnect gracefully
            MPDev.disconnectMPDev()
            Cursor.Show()
        End Sub
        Private Function PreLoadImageStim(ByVal imageList As ArrayList) As Boolean
            imageList.TrimToSize()
            'shuffle the file name list
            ShuffleImageList(imageList)

            Dim count As Integer = imageList.Count
            Dim c As Integer

            'create a stimulus report
            StimReporter = New StimulusReport(count - 1, OutputDir)

            Dim str As String
            Dim uri As Uri
            For c = 0 To count - 1
                str = imageList(c)
                uri = New Uri(str)
                'initiate the stimulist in the stimreporter
                StimReporter.StimulusList(c).Score = 0
                StimReporter.StimulusList(c).Location = uri.AbsoluteUri
                StimReporter.StimulusList(c).UniqueID = (str.GetHashCode() + System.DateTime.Now.Ticks).ToString()
                StimReporter.StimulusList(c).Waveform = New ArrayList
                StimReporter.StimulusList(c).theImage = New Bitmap(str)
                StimReporter.StimulusList(c).theWaveImage = New Bitmap(1024, 200)
            Next

            Return True
        End Function
        Private Function ConfigureAcquisition() As Boolean
            Dim retval As MPDev.MPRETURNCODE

            'connect to the MP36
            retval = MPDev.connectMPDev(MPDev.MPTYPE.MP36, MPDev.MPCOMTYPE.MPUSB, "")

            If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
                Return False
            End If

            'set analog channels
            retval = MPDev.setAcqChannels(AcqConfig.aCH(0))

            If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
                Return False
            End If

            'set digital channels
            retval = MPDev.setDigitalAcqChannels(AcqConfig.dCH(0))

            If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
                Return False
            End If

            'set sample rate
            retval = MPDev.setSampleRate(1000.0 / AcqConfig.frequency)

            If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
                Return False
            End If

            'prompt the user to find the channel preset xml file
            Dim presetXMLFileDialog As OpenFileDialog = New OpenFileDialog
            presetXMLFileDialog.ValidateNames = True
            presetXMLFileDialog.Filter = "XML File (*.xml)|*.xml"
            presetXMLFileDialog.Title = "Select the Channel Presets XML File"

            Dim filename As String = ""

            Cursor.Show()
            'loop unitl a valid channel preset xml file has been loaded
            Do
                filename = ""

                If (presetXMLFileDialog.ShowDialog() = DialogResult.OK) Then
                    filename = presetXMLFileDialog.FileName
                End If

                'load the preset file
                retval = MPDev.loadXMLPresetFile(filename)
            Loop While Not (retval = MPDev.MPRETURNCODE.MPSUCCESS)  'if the preset file is not valid prompt again
            Cursor.Hide()

            'apply ecg preset to analog channel 1
            retval = MPDev.configChannelByPresetID(0, "a102")

            If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
                Return False
            End If

            Return True
        End Function
        Private Sub ClearScreen()
            'clear the screan by repainting it with the background color
            CreateGraphics().Clear(BackColor)
        End Sub
        Private Sub GetAnalogChannelZero(ByRef waveform As ArrayList, ByRef data() As Double, ByVal buffSize As Integer)
            Dim i As Integer
            Dim index As Integer

            'go through the buffer
            For i = 0 To buffSize
                index = i Mod 9

                'if the index corresponds to a sample point for 
                'Channel Zero then add it to the the waveform
                If index = 0 Then
                    waveform.Add(data(i))
                End If
            Next
        End Sub

        Private Function GetScore(ByRef data() As Double, ByVal buffSize As Integer) As Integer
            Dim i As Integer
            Dim index As Integer
            Dim val As Double

            'assumes the last eight channels are the digital channels
            'assume that the user is not pressing more than one button at a time

            'and there's only one analog channel (nine active channels)
            'score is 1 through 8

            'find the last response to the image
            For i = buffSize To 0 Step -1
                val = data(i)
                index = i Mod 9

                If index > 0 And val < 4.9 Then
                    Return index
                End If
            Next

            Return 0

        End Function

        Private Sub ShuffleImageList(ByVal imageList As ArrayList)
            Dim randGenerator As Random = New Random
            Dim temp As String
            Dim repeat As Integer
            Dim A As Integer
            Dim B As Integer

            'shuffle the image
            For repeat = 0 To imageList.Count * 2
                A = randGenerator.Next() Mod imageList.Count
                B = randGenerator.Next() Mod imageList.Count
                temp = imageList(A)
                imageList(A) = imageList(B)
                imageList(B) = temp
            Next
        End Sub
        Private Sub DisplayImage(ByVal index As Integer)
            Dim g As Graphics = Me.CreateGraphics()
            Dim picture As Bitmap = StimReporter.StimulusList(index).theImage
            Dim windowRatio As Double = Size.Width / Size.Height
            Dim picRatio As Double = picture.Width / picture.Height
            Dim location As Point
            Dim dimension As Size

            'displays the image in the center of the screen
            dimension.Width = picture.Width
            dimension.Height = picture.Height
            location.X = 0
            location.Y = 0

            'fit to screen algorithm
            'shrink bigger image and just center the smaller images
            If dimension.Width > Size.Width Or dimension.Height > Size.Height Then
                If windowRatio > picRatio Then
                    dimension.Height = Size.Height
                    dimension.Width = (Size.Height * picture.Width) / picture.Height
                ElseIf picRatio < windowRatio Then
                    dimension.Height = (Size.Width * picture.Height) / picture.Width
                    dimension.Width = Size.Width
                Else
                    dimension.Height = (Size.Width * picture.Height) / picture.Width
                    dimension.Width = (Size.Height * picture.Width) / picture.Height
                End If
            End If

            location.X = (Size.Width - dimension.Width) / 2
            location.Y = (Size.Height - dimension.Height) / 2

            'darw the image
            g.DrawImage(picture, location.X, location.Y, dimension.Width, dimension.Height)

        End Sub
        Private Sub StimuliDisplay_Click(ByVal sender As Object, ByVal e As System.Windows.Forms.MouseEventArgs) Handles MyBase.MouseDown
            'if the user presses the left click button exit
            If e.Button = MouseButtons.Left Then
                QUIT = True
                Close()
            End If
        End Sub
        Private Sub StimuliDisplay_KeyDown(ByVal sender As Object, ByVal e As System.Windows.Forms.KeyEventArgs) Handles MyBase.KeyDown
            'if the user presses the escape or F11 exit
            Select Case (e.KeyData)
                Case Keys.Escape
                    QUIT = True
                    Close()
                Case Keys.F11
                    QUIT = True
                    Close()
            End Select
        End Sub
        Private Sub StimuliDisplay_Closed(ByVal sender As Object, ByVal e As System.EventArgs) Handles MyBase.Closed
            'if the user closes the display window abort
            Abort()
        End Sub
    End Class
End Namespace

