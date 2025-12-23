''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
' Copyright 2004-2023 BIOPAC Systems, Inc.
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
' Portions Copyright 2004-2023 BIOPAC Systems, Inc.
'
' 2. Altered source versions must be plainly marked as such, and must not be 
' misrepresented as being the original software.
'
' 3. This notice may not be removed or altered from any source distribution.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

Imports System.IO
Imports System.Text
Imports Microsoft.VisualBasic
Imports System.Reflection
Imports System.CodeDom.Compiler

Public Class FunctionGenerator
    Inherits System.Windows.Forms.Form

    Dim Calculator As Object    'for the dynamically created object
    Dim calcMeth As MethodInfo  'for accessing the method in the dynamically created object
    Dim start As Long           'for keeping track of the start time
    Dim doubleVal(0) As Double  'for holding the actual values in a file
    Dim sample As Long = 0      'for keeping track of the number of samples in a file
    Dim outch As Long = 0       'for keeping track which Analog Channel to output to


#Region " Windows Form Designer generated code "

    Public Sub New()
        MyBase.New()

        'This call is required by the Windows Form Designer.
        InitializeComponent()

        'Add any initialization after the InitializeComponent() call

    End Sub

    'Form overrides dispose to clean up the component list.
    Protected Overloads Overrides Sub Dispose(ByVal disposing As Boolean)
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
    Friend WithEvents GroupParam As System.Windows.Forms.GroupBox
    Friend WithEvents RadioA0 As System.Windows.Forms.RadioButton
    Friend WithEvents RadioA1 As System.Windows.Forms.RadioButton
    Friend WithEvents RadioON As System.Windows.Forms.RadioButton
    Friend WithEvents RadioOFF As System.Windows.Forms.RadioButton
    Friend WithEvents TextInterval As System.Windows.Forms.TextBox
    Friend WithEvents Label1 As System.Windows.Forms.Label
    Friend WithEvents Label2 As System.Windows.Forms.Label
    Friend WithEvents RichTextBox1 As System.Windows.Forms.RichTextBox
    Friend WithEvents ButtonSine As System.Windows.Forms.Button
    Friend WithEvents ButtonStart As System.Windows.Forms.Button
    Friend WithEvents GroupPredefun As System.Windows.Forms.GroupBox
    Friend WithEvents GroupExt As System.Windows.Forms.GroupBox
    Friend WithEvents GroupOutCh As System.Windows.Forms.GroupBox
    Friend WithEvents GroupOutput As System.Windows.Forms.GroupBox
    Friend WithEvents ButtonSQUARE As System.Windows.Forms.Button
    Friend WithEvents ButtonFile As System.Windows.Forms.Button
    Friend WithEvents ButtonCLEAR As System.Windows.Forms.Button
    Friend WithEvents OpenFileDialog1 As System.Windows.Forms.OpenFileDialog
    Friend WithEvents TimerDelay1 As System.Windows.Forms.Timer
    Friend WithEvents TimerDelay2 As System.Windows.Forms.Timer
    <System.Diagnostics.DebuggerStepThrough()> Private Sub InitializeComponent()
        Me.components = New System.ComponentModel.Container
        Me.GroupParam = New System.Windows.Forms.GroupBox
        Me.ButtonCLEAR = New System.Windows.Forms.Button
        Me.ButtonStart = New System.Windows.Forms.Button
        Me.Label2 = New System.Windows.Forms.Label
        Me.Label1 = New System.Windows.Forms.Label
        Me.TextInterval = New System.Windows.Forms.TextBox
        Me.GroupExt = New System.Windows.Forms.GroupBox
        Me.RadioOFF = New System.Windows.Forms.RadioButton
        Me.RadioON = New System.Windows.Forms.RadioButton
        Me.GroupOutCh = New System.Windows.Forms.GroupBox
        Me.RadioA1 = New System.Windows.Forms.RadioButton
        Me.RadioA0 = New System.Windows.Forms.RadioButton
        Me.GroupPredefun = New System.Windows.Forms.GroupBox
        Me.ButtonFile = New System.Windows.Forms.Button
        Me.ButtonSQUARE = New System.Windows.Forms.Button
        Me.ButtonSine = New System.Windows.Forms.Button
        Me.GroupOutput = New System.Windows.Forms.GroupBox
        Me.RichTextBox1 = New System.Windows.Forms.RichTextBox
        Me.TimerDelay1 = New System.Windows.Forms.Timer(Me.components)
        Me.OpenFileDialog1 = New System.Windows.Forms.OpenFileDialog
        Me.TimerDelay2 = New System.Windows.Forms.Timer(Me.components)
        Me.GroupParam.SuspendLayout()
        Me.GroupExt.SuspendLayout()
        Me.GroupOutCh.SuspendLayout()
        Me.GroupPredefun.SuspendLayout()
        Me.GroupOutput.SuspendLayout()
        Me.SuspendLayout()
        '
        'GroupParam
        '
        Me.GroupParam.Controls.Add(Me.ButtonCLEAR)
        Me.GroupParam.Controls.Add(Me.ButtonStart)
        Me.GroupParam.Controls.Add(Me.Label2)
        Me.GroupParam.Controls.Add(Me.Label1)
        Me.GroupParam.Controls.Add(Me.TextInterval)
        Me.GroupParam.Controls.Add(Me.GroupExt)
        Me.GroupParam.Controls.Add(Me.GroupOutCh)
        Me.GroupParam.Location = New System.Drawing.Point(8, 8)
        Me.GroupParam.Name = "GroupParam"
        Me.GroupParam.Size = New System.Drawing.Size(416, 112)
        Me.GroupParam.TabIndex = 0
        Me.GroupParam.TabStop = False
        Me.GroupParam.Text = "Parameters"
        '
        'ButtonCLEAR
        '
        Me.ButtonCLEAR.Location = New System.Drawing.Point(328, 56)
        Me.ButtonCLEAR.Name = "ButtonCLEAR"
        Me.ButtonCLEAR.Size = New System.Drawing.Size(72, 48)
        Me.ButtonCLEAR.TabIndex = 6
        Me.ButtonCLEAR.Text = "CLEAR"
        '
        'ButtonStart
        '
        Me.ButtonStart.Location = New System.Drawing.Point(240, 56)
        Me.ButtonStart.Name = "ButtonStart"
        Me.ButtonStart.Size = New System.Drawing.Size(88, 48)
        Me.ButtonStart.TabIndex = 5
        Me.ButtonStart.Text = "START"
        '
        'Label2
        '
        Me.Label2.Location = New System.Drawing.Point(384, 24)
        Me.Label2.Name = "Label2"
        Me.Label2.Size = New System.Drawing.Size(24, 24)
        Me.Label2.TabIndex = 4
        Me.Label2.Text = "ms"
        '
        'Label1
        '
        Me.Label1.Location = New System.Drawing.Point(224, 24)
        Me.Label1.Name = "Label1"
        Me.Label1.Size = New System.Drawing.Size(72, 24)
        Me.Label1.TabIndex = 3
        Me.Label1.Text = "Output every"
        '
        'TextInterval
        '
        Me.TextInterval.Location = New System.Drawing.Point(296, 24)
        Me.TextInterval.Name = "TextInterval"
        Me.TextInterval.Size = New System.Drawing.Size(80, 20)
        Me.TextInterval.TabIndex = 2
        Me.TextInterval.Text = "50"
        '
        'GroupExt
        '
        Me.GroupExt.Controls.Add(Me.RadioOFF)
        Me.GroupExt.Controls.Add(Me.RadioON)
        Me.GroupExt.Location = New System.Drawing.Point(112, 16)
        Me.GroupExt.Name = "GroupExt"
        Me.GroupExt.Size = New System.Drawing.Size(104, 88)
        Me.GroupExt.TabIndex = 1
        Me.GroupExt.TabStop = False
        Me.GroupExt.Text = "External Trigger"
        '
        'RadioOFF
        '
        Me.RadioOFF.Checked = True
        Me.RadioOFF.Location = New System.Drawing.Point(16, 56)
        Me.RadioOFF.Name = "RadioOFF"
        Me.RadioOFF.Size = New System.Drawing.Size(48, 16)
        Me.RadioOFF.TabIndex = 1
        Me.RadioOFF.TabStop = True
        Me.RadioOFF.Text = "OFF"
        '
        'RadioON
        '
        Me.RadioON.Location = New System.Drawing.Point(16, 24)
        Me.RadioON.Name = "RadioON"
        Me.RadioON.Size = New System.Drawing.Size(40, 24)
        Me.RadioON.TabIndex = 0
        Me.RadioON.Text = "ON"
        '
        'GroupOutCh
        '
        Me.GroupOutCh.Controls.Add(Me.RadioA1)
        Me.GroupOutCh.Controls.Add(Me.RadioA0)
        Me.GroupOutCh.Location = New System.Drawing.Point(16, 16)
        Me.GroupOutCh.Name = "GroupOutCh"
        Me.GroupOutCh.Size = New System.Drawing.Size(88, 88)
        Me.GroupOutCh.TabIndex = 0
        Me.GroupOutCh.TabStop = False
        Me.GroupOutCh.Text = "Output Using"
        '
        'RadioA1
        '
        Me.RadioA1.Location = New System.Drawing.Point(8, 56)
        Me.RadioA1.Name = "RadioA1"
        Me.RadioA1.Size = New System.Drawing.Size(40, 24)
        Me.RadioA1.TabIndex = 1
        Me.RadioA1.Text = "A1"
        '
        'RadioA0
        '
        Me.RadioA0.Checked = True
        Me.RadioA0.Location = New System.Drawing.Point(8, 24)
        Me.RadioA0.Name = "RadioA0"
        Me.RadioA0.Size = New System.Drawing.Size(40, 24)
        Me.RadioA0.TabIndex = 0
        Me.RadioA0.TabStop = True
        Me.RadioA0.Text = "A0"
        '
        'GroupPredefun
        '
        Me.GroupPredefun.Controls.Add(Me.ButtonFile)
        Me.GroupPredefun.Controls.Add(Me.ButtonSQUARE)
        Me.GroupPredefun.Controls.Add(Me.ButtonSine)
        Me.GroupPredefun.Location = New System.Drawing.Point(440, 8)
        Me.GroupPredefun.Name = "GroupPredefun"
        Me.GroupPredefun.Size = New System.Drawing.Size(128, 112)
        Me.GroupPredefun.TabIndex = 1
        Me.GroupPredefun.TabStop = False
        Me.GroupPredefun.Text = "Predifined Functions"
        '
        'ButtonFile
        '
        Me.ButtonFile.Location = New System.Drawing.Point(8, 72)
        Me.ButtonFile.Name = "ButtonFile"
        Me.ButtonFile.Size = New System.Drawing.Size(112, 24)
        Me.ButtonFile.TabIndex = 3
        Me.ButtonFile.Text = "FILE"
        '
        'ButtonSQUARE
        '
        Me.ButtonSQUARE.Location = New System.Drawing.Point(8, 48)
        Me.ButtonSQUARE.Name = "ButtonSQUARE"
        Me.ButtonSQUARE.Size = New System.Drawing.Size(112, 24)
        Me.ButtonSQUARE.TabIndex = 2
        Me.ButtonSQUARE.Text = "SQUARE"
        '
        'ButtonSine
        '
        Me.ButtonSine.Location = New System.Drawing.Point(8, 24)
        Me.ButtonSine.Name = "ButtonSine"
        Me.ButtonSine.Size = New System.Drawing.Size(112, 24)
        Me.ButtonSine.TabIndex = 0
        Me.ButtonSine.Text = "SINE"
        '
        'GroupOutput
        '
        Me.GroupOutput.Controls.Add(Me.RichTextBox1)
        Me.GroupOutput.Location = New System.Drawing.Point(8, 128)
        Me.GroupOutput.Name = "GroupOutput"
        Me.GroupOutput.Size = New System.Drawing.Size(560, 192)
        Me.GroupOutput.TabIndex = 2
        Me.GroupOutput.TabStop = False
        Me.GroupOutput.Text = "Function"
        '
        'RichTextBox1
        '
        Me.RichTextBox1.Font = New System.Drawing.Font("Verdana", 12.0!, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, CType(0, Byte))
        Me.RichTextBox1.Location = New System.Drawing.Point(8, 24)
        Me.RichTextBox1.Name = "RichTextBox1"
        Me.RichTextBox1.Size = New System.Drawing.Size(544, 160)
        Me.RichTextBox1.TabIndex = 0
        Me.RichTextBox1.Text = ""
        '
        'TimerDelay1
        '
        '
        'TimerDelay2
        '
        '
        'FunctionGenerator
        '
        Me.AutoScaleBaseSize = New System.Drawing.Size(5, 13)
        Me.ClientSize = New System.Drawing.Size(576, 325)
        Me.Controls.Add(Me.GroupOutput)
        Me.Controls.Add(Me.GroupPredefun)
        Me.Controls.Add(Me.GroupParam)
        Me.Name = "FunctionGenerator"
        Me.Text = "Function Generator"
        Me.GroupParam.ResumeLayout(False)
        Me.GroupExt.ResumeLayout(False)
        Me.GroupOutCh.ResumeLayout(False)
        Me.GroupPredefun.ResumeLayout(False)
        Me.GroupOutput.ResumeLayout(False)
        Me.ResumeLayout(False)

    End Sub

#End Region

    'handles starts and stop button logic
    Private Sub ButtonStart_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles ButtonStart.Click

        'make sure there's at least a non whitespace character in the text box
        If Me.RichTextBox1.Text.Trim = "" Then
            MessageBox.Show("Function can't be blank", "Function Format Error", MessageBoxButtons.OK, MessageBoxIcon.Exclamation)
            Return
        End If

        'disable buttons and group boxes
        Me.ButtonStart.Enabled = False
        Me.RichTextBox1.Enabled = False
        Me.GroupParam.Enabled = False
        Me.GroupPredefun.Enabled = False

        'logic to stop the function generator
        If Me.ButtonStart.Text = "STOP" Then
            Me.TimerDelay1.Stop()   'stop timer
            Me.TimerDelay2.Stop()   'stop timer
            MPDev.disconnectMPDev() 'disconnect from mp device
            Me.ButtonStart.Text = "START"
            Me.ButtonStart.Enabled = True
            Me.GroupParam.Enabled = True
            Me.GroupPredefun.Enabled = True
            Me.GroupOutCh.Enabled = True
            Me.GroupExt.Enabled = True
            Me.TextInterval.Enabled = True
            Me.ButtonCLEAR.Enabled = True

            If Not (Me.RichTextBox1.Text.StartsWith("FILE=")) Then
                Me.RichTextBox1.Enabled = True
            End If

            Return
        End If

        'prepare to output equation or file
        If Me.RichTextBox1.Text.StartsWith("FILE=") Then
            'prepare the file
            If Not (PrepareFile()) Then
                Return
            End If
        Else
            'prepare the function
            If Not (PrepareFunction()) Then
                Return
            End If
        End If

        'prepare the mp device
        If Not (PrepareMPDevice()) Then
            If Me.RichTextBox1.Text.StartsWith("FILE=") Then
                Me.RichTextBox1.Enabled = False
            End If
            Return
        End If

        'set which Analog channel to output to
        If Me.RadioA0.Checked Then
            Me.outch = 0
        Else
            Me.outch = 1
        End If

        'see if user want to externally trigger
        If Me.RadioON.Checked Then
            Me.ButtonStart.Text = "TRIGGER"
            Me.ButtonStart.Refresh()
            'wait for the trigger to arrive
            If Not (WaitforTrigger()) Then
                Me.TimerDelay1.Stop()
                Me.TimerDelay2.Stop()
                MPDev.disconnectMPDev()
                Me.ButtonStart.Text = "START"
                Me.ButtonStart.Enabled = True
                Me.GroupParam.Enabled = True
                Me.GroupPredefun.Enabled = True
                Me.GroupOutCh.Enabled = True
                Me.GroupExt.Enabled = True
                Me.TextInterval.Enabled = True
                Me.ButtonCLEAR.Enabled = True

                If Not (Me.RichTextBox1.Text.StartsWith("FILE=")) Then
                    Me.RichTextBox1.Enabled = True
                End If

                Return
            End If
            Me.ButtonStart.Text = "STOP"
            Me.ButtonStart.Refresh()
        End If

        'Start the timer for outputing the file or a function
        If Me.RichTextBox1.Text.StartsWith("FILE=") Then
            Me.TimerDelay2.Interval = CInt(System.Double.Parse(Me.TextInterval.Text))
            Me.TimerDelay2.Start()
        Else
            Me.TimerDelay1.Interval = CInt(System.Double.Parse(Me.TextInterval.Text))
            Me.TimerDelay1.Start()
        End If

        Me.ButtonStart.Text = "STOP"
        'save the time the function was started
        start = System.DateTime.Now.Ticks

        Me.GroupParam.Enabled = True
        Me.GroupOutCh.Enabled = False
        Me.GroupExt.Enabled = False
        Me.TextInterval.Enabled = False
        Me.ButtonCLEAR.Enabled = False
        Me.ButtonStart.Enabled = True
    End Sub

    'blocks until the MP device receives the external trigger
    Private Function WaitforTrigger() As Boolean
        Dim rval As MPDev.MPRETURNCODE
        Dim dum As Integer
        Dim ach(15) As Boolean
        Dim data As Double

        'SETUP A DUMMY ACQUISITION
        rval = MPDev.setSampleRate(1000.0) ' sample 1 hz

        If Not (rval = MPDev.MPRETURNCODE.MPSUCCESS) Then
            Return False
        End If

        ach(0) = True 'at least one active channel

        rval = MPDev.setAcqChannels(ach(0)) 'set channels

        If Not (rval = MPDev.MPRETURNCODE.MPSUCCESS) Then
            Return False
        End If

        rval = MPDev.setMPTrigger(MPDev.TRIGGEROPT.MPTRIGEXT, True, 0.0, 1) 'set trigger positive edge

        If Not (rval = MPDev.MPRETURNCODE.MPSUCCESS) Then
            Return False
        End If

        rval = MPDev.startMPAcqDaemon() 'start Acquisition daemon

        If Not (rval = MPDev.MPRETURNCODE.MPSUCCESS) Then
            Return False
        End If

        rval = MPDev.startAcquisition() 'start Acquisition

        If Not (rval = MPDev.MPRETURNCODE.MPSUCCESS) Then
            Return False
        End If


        'this function will block until the first sample is acquired
        'which implies the trigger has been actiavated
        MPDev.receiveMPData(data, 1, dum)

        'stop acquisition because the daemon is not needed anymore
        MPDev.stopAcquisition()

        Return True
    End Function

    'prepare a file with double values one per line
    Private Function PrepareFile() As Boolean
        Dim path As String = Me.RichTextBox1.Text.Replace("FILE=", "")
        Dim text As String = File.OpenText(path).ReadToEnd()
        Dim len As Long = 0
        Dim index As Long = 0
        Dim filevalue() As String = text.Split(vbCrLf)

        'ignore lines with white space
        For Each strVal As String In filevalue
            If Not (strVal.Trim() = "") Then
                len = len + 1
            End If
        Next

        'allocate array
        ReDim doubleVal(len - 1)

        'store values in the array
        For Each strVal As String In filevalue
            If Not (strVal.Trim() = "") Then
                doubleVal.SetValue(System.Double.Parse(strVal.Trim()), index)
                index = index + 1
            End If
        Next

        'reset sample index to zero
        sample = 0

        Return True
    End Function

    'prepare function that gets dynamically compiled in memory
    Private Function PrepareFunction() As Boolean
        'This function dynamically creates and compiles this class:
        '           Imports System.Math
        '           Public Class Calc
        '               Public Function Calculate(ByVal x As Double) As Double
        '               <CODE>
        '               End Function
        '           End Class
        'where <CODE> gets replaced by the function in the text box


        Dim code As String
        code = "Imports System.Math" + vbCrLf + "Public Class Calc" + vbCrLf + "Public Function Calculate(ByVal x As Double) As Double" + vbCrLf + "<CODE>" + vbCrLf + "End Function" + vbCrLf + "End Class"
        'replace <CODE> with actual code
        code = code.Replace("<CODE>", "Return " + Me.RichTextBox1.Text.Trim())

        'Create a compiler
        Dim provider As New VBCodeProvider
        Dim iccomp As ICodeCompiler = provider.CreateCompiler()
        Dim compara As New CompilerParameters

        'add references
        compara.ReferencedAssemblies.Add("system.dll")
        compara.ReferencedAssemblies.Add("system.data.dll")
        compara.ReferencedAssemblies.Add("system.xml.dll")
        'configure compiler
        compara.GenerateExecutable = False
        compara.GenerateInMemory = True

        'compile 
        Dim compres As CompilerResults = iccomp.CompileAssemblyFromSource(compara, code)

        'if theres an error return the error error
        If compres.Errors.HasErrors Then
            Dim err As New StringBuilder

            err.Append("Errors: " + vbCrLf)
            For Each cerr As CompilerError In compres.Errors
                err.AppendFormat("{0}" + vbCrLf, cerr.ErrorText)
            Next cerr

            MessageBox.Show(err.ToString(), "Function Format Error", MessageBoxButtons.OK, MessageBoxIcon.Exclamation)
            Me.ButtonStart.Text = "START"
            Me.ButtonStart.Enabled = True
            Me.RichTextBox1.Enabled = True
            Me.GroupParam.Enabled = True
            Me.GroupPredefun.Enabled = True
            Return False
        End If

        'create an instance of the dynamically object
        Me.Calculator = compres.CompiledAssembly.CreateInstance("Calc")
        'assign method
        Me.calcMeth = Calculator.GetType().GetMethod("Calculate")

        Return True

    End Function

    'prepare mp device by connecting to it
    Private Function PrepareMPDevice() As Boolean
        Dim rval As MPDev.MPRETURNCODE

        'connect to the MP Device
        'BHAPI 1.1 supports MP160 auto discovery. 
        rval = MPDev.connectMPDev(MPDev.MPTYPE.MP160, MPDev.MPCOMTYPE.MPUDP, "auto")

        If Not (rval = MPDev.MPRETURNCODE.MPSUCCESS) Then
            MPDev.disconnectMPDev()
            MessageBox.Show("Failed to connect to MP Device: " + rval.ToString() + vbCrLf, "MP Communication Error", MessageBoxButtons.OK, MessageBoxIcon.Exclamation)
            Me.ButtonStart.Text = "START"
            Me.ButtonStart.Enabled = True
            Me.ButtonStart.Enabled = True
            Me.RichTextBox1.Enabled = True
            Me.GroupParam.Enabled = True
            Me.GroupPredefun.Enabled = True

            Return False
        End If

        Return True
    End Function

    'handles the tick event of the timer for outputing a function
    Private Sub TimerDelay1_Tick(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TimerDelay1.Tick
        TimerDelay1.Enabled = False
        Dim param(0) As Object
        Dim retval As MPDev.MPRETURNCODE
        Dim outval As Double

        'call Calculator method in the dynamically compile object
        param(0) = ((System.DateTime.Now.Ticks - start) * 100) / 1000000
        outval = Me.calcMeth.Invoke(Calculator, param)
        retval = MPDev.setAnalogOut(outval, outch)

        'communication error
        If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
            MessageBox.Show("Failed to Set Analog Out", "MP Communication Error", MessageBoxButtons.OK, MessageBoxIcon.Exclamation)
            Me.TimerDelay1.Stop()
            MPDev.disconnectMPDev()
            Me.ButtonStart.Text = "START"
            Me.ButtonStart.Enabled = True
            Me.GroupParam.Enabled = True
            Me.GroupPredefun.Enabled = True
            Me.GroupOutCh.Enabled = True
            Me.GroupExt.Enabled = True
            Me.TextInterval.Enabled = True
            Me.ButtonCLEAR.Enabled = True
            Me.RichTextBox1.Enabled = True
            Return
        End If

        TimerDelay1.Enabled = True
    End Sub

    'handles the tick event of the timer for outputing a set of values in a file
    Private Sub TimerDelay2_Tick(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles TimerDelay2.Tick
        TimerDelay2.Enabled = False
        Dim retval As MPDev.MPRETURNCODE
        Dim outval As Double

        'output a sample each tick, wrap around at the end
        sample = sample + 1
        outval = Me.doubleVal(sample Mod doubleVal.Length)
        retval = MPDev.setAnalogOut(outval, outch)

        'communication error
        If Not (retval = MPDev.MPRETURNCODE.MPSUCCESS) Then
            MessageBox.Show("Failed to Set Analog Out", "MP Communication Error", MessageBoxButtons.OK, MessageBoxIcon.Exclamation)
            Me.TimerDelay2.Stop()
            MPDev.disconnectMPDev()
            Me.ButtonStart.Text = "START"
            Me.ButtonStart.Enabled = True
            Me.GroupParam.Enabled = True
            Me.GroupPredefun.Enabled = True
            Me.GroupOutCh.Enabled = True
            Me.GroupExt.Enabled = True
            Me.TextInterval.Enabled = True
            Me.ButtonCLEAR.Enabled = True
            Me.RichTextBox1.Enabled = True
            Return
        End If

        TimerDelay2.Enabled = True
    End Sub

    'insert sine wave
    Private Sub ButtonSine_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles ButtonSine.Click
        Me.RichTextBox1.Text += "(1*Sin(x))"
    End Sub

    'insert square wave
    Private Sub ButtonSquare_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles ButtonSQUARE.Click
        Me.RichTextBox1.Text += "Sign( Sin ( (2 * PI * x) / .0025) )"
    End Sub

    'clear text box
    Private Sub ButtonCLEAR_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles ButtonCLEAR.Click
        Me.RichTextBox1.Text = ""
        Me.RichTextBox1.Enabled = True
    End Sub

    'open dialog box
    Private Sub ButtonFile_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles ButtonFile.Click
        Me.OpenFileDialog1 = New OpenFileDialog

        Me.OpenFileDialog1.Filter = "TXT .txt|*.txt"
        Me.OpenFileDialog1.Title = "Line Delimeted Values"
        Me.OpenFileDialog1.CheckFileExists = True
        Me.OpenFileDialog1.Multiselect = False

        If Me.OpenFileDialog1.ShowDialog() = DialogResult.OK Then
            If Not (File.Exists(Me.OpenFileDialog1.FileName)) Then
                MessageBox.Show("File does not exist", "File Error", MessageBoxButtons.OK, MessageBoxIcon.Exclamation)
                Return
            End If

            Me.RichTextBox1.Text = "FILE=" + Me.OpenFileDialog1.FileName
            Me.RichTextBox1.Enabled = False
        End If

    End Sub

    'make sure to disconnect from the mp device before exit
    Private Sub FormExit(ByVal sender As Object, ByVal e As System.ComponentModel.CancelEventArgs) Handles MyBase.Closing
        MPDev.disconnectMPDev()
    End Sub

End Class
