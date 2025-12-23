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

Imports System
Imports System.IO

Namespace Biopac.API.MPDevice
    Public Class ImageStimGUI
        Inherits System.Windows.Forms.Form

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
        Friend WithEvents goButton As System.Windows.Forms.Button
        Friend WithEvents inputDirGroupBox As System.Windows.Forms.GroupBox
        Friend WithEvents inputDirButton As System.Windows.Forms.Button
        Friend WithEvents inputDirTextBox As System.Windows.Forms.TextBox
        Friend WithEvents outputDirButton As System.Windows.Forms.Button
        Friend WithEvents outputDirTextBox As System.Windows.Forms.TextBox
        Friend WithEvents inputFolderBrowserDialog As System.Windows.Forms.FolderBrowserDialog
        Friend WithEvents outputFolderBrowserDialog As System.Windows.Forms.FolderBrowserDialog
        Friend WithEvents outputGroupBox As System.Windows.Forms.GroupBox
        <System.Diagnostics.DebuggerStepThrough()> Private Sub InitializeComponent()
            Me.inputDirGroupBox = New System.Windows.Forms.GroupBox
            Me.inputDirButton = New System.Windows.Forms.Button
            Me.inputDirTextBox = New System.Windows.Forms.TextBox
            Me.inputFolderBrowserDialog = New System.Windows.Forms.FolderBrowserDialog
            Me.goButton = New System.Windows.Forms.Button
            Me.outputGroupBox = New System.Windows.Forms.GroupBox
            Me.outputDirButton = New System.Windows.Forms.Button
            Me.outputDirTextBox = New System.Windows.Forms.TextBox
            Me.outputFolderBrowserDialog = New System.Windows.Forms.FolderBrowserDialog
            Me.inputDirGroupBox.SuspendLayout()
            Me.outputGroupBox.SuspendLayout()
            Me.SuspendLayout()
            '
            'inputDirGroupBox
            '
            Me.inputDirGroupBox.Controls.Add(Me.inputDirButton)
            Me.inputDirGroupBox.Controls.Add(Me.inputDirTextBox)
            Me.inputDirGroupBox.Location = New System.Drawing.Point(7, 8)
            Me.inputDirGroupBox.Name = "inputDirGroupBox"
            Me.inputDirGroupBox.Size = New System.Drawing.Size(376, 40)
            Me.inputDirGroupBox.TabIndex = 1
            Me.inputDirGroupBox.TabStop = False
            Me.inputDirGroupBox.Text = "Select Image Directory"
            '
            'inputDirButton
            '
            Me.inputDirButton.Location = New System.Drawing.Point(336, 16)
            Me.inputDirButton.Name = "inputDirButton"
            Me.inputDirButton.Size = New System.Drawing.Size(32, 16)
            Me.inputDirButton.TabIndex = 1
            Me.inputDirButton.Text = "..."
            '
            'inputDirTextBox
            '
            Me.inputDirTextBox.Location = New System.Drawing.Point(8, 16)
            Me.inputDirTextBox.Name = "inputDirTextBox"
            Me.inputDirTextBox.ReadOnly = True
            Me.inputDirTextBox.Size = New System.Drawing.Size(320, 20)
            Me.inputDirTextBox.TabIndex = 0
            Me.inputDirTextBox.Text = ""
            '
            'inputFolderBrowserDialog
            '
            Me.inputFolderBrowserDialog.Description = "Select Folder with Image Files"
            Me.inputFolderBrowserDialog.ShowNewFolderButton = False
            '
            'goButton
            '
            Me.goButton.Enabled = False
            Me.goButton.Location = New System.Drawing.Point(7, 96)
            Me.goButton.Name = "goButton"
            Me.goButton.Size = New System.Drawing.Size(376, 24)
            Me.goButton.TabIndex = 3
            Me.goButton.Text = "GO!!!"
            '
            'outputGroupBox
            '
            Me.outputGroupBox.Controls.Add(Me.outputDirButton)
            Me.outputGroupBox.Controls.Add(Me.outputDirTextBox)
            Me.outputGroupBox.Enabled = False
            Me.outputGroupBox.Location = New System.Drawing.Point(7, 48)
            Me.outputGroupBox.Name = "outputGroupBox"
            Me.outputGroupBox.Size = New System.Drawing.Size(376, 40)
            Me.outputGroupBox.TabIndex = 4
            Me.outputGroupBox.TabStop = False
            Me.outputGroupBox.Text = "Select Output Directory"
            '
            'outputDirButton
            '
            Me.outputDirButton.Location = New System.Drawing.Point(336, 16)
            Me.outputDirButton.Name = "outputDirButton"
            Me.outputDirButton.Size = New System.Drawing.Size(32, 16)
            Me.outputDirButton.TabIndex = 1
            Me.outputDirButton.Text = "..."
            '
            'outputDirTextBox
            '
            Me.outputDirTextBox.Location = New System.Drawing.Point(8, 16)
            Me.outputDirTextBox.Name = "outputDirTextBox"
            Me.outputDirTextBox.ReadOnly = True
            Me.outputDirTextBox.Size = New System.Drawing.Size(320, 20)
            Me.outputDirTextBox.TabIndex = 0
            Me.outputDirTextBox.Text = ""
            '
            'outputFolderBrowserDialog
            '
            Me.outputFolderBrowserDialog.Description = "Select Output Folder"
            '
            'ImageStimGUI
            '
            Me.AutoScaleBaseSize = New System.Drawing.Size(5, 13)
            Me.ClientSize = New System.Drawing.Size(390, 126)
            Me.Controls.Add(Me.outputGroupBox)
            Me.Controls.Add(Me.goButton)
            Me.Controls.Add(Me.inputDirGroupBox)
            Me.MaximizeBox = False
            Me.MinimizeBox = False
            Me.Name = "ImageStimGUI"
            Me.Text = "Image Stim"
            Me.inputDirGroupBox.ResumeLayout(False)
            Me.outputGroupBox.ResumeLayout(False)
            Me.ResumeLayout(False)

        End Sub

#End Region

        Private Sub inputDirButton_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles inputDirButton.Click
            goButton.Enabled = False
            outputGroupBox.Enabled = False

            'let the user browse for a folder which contain the images
            If inputFolderBrowserDialog.ShowDialog() = DialogResult.OK Then
                inputDirTextBox.Text = inputFolderBrowserDialog.SelectedPath
                outputGroupBox.Enabled = True
            End If
        End Sub

        Private Sub outputDirButton_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles outputDirButton.Click
            goButton.Enabled = False

            'let the user browse for a folder where to output the data files
            If outputFolderBrowserDialog.ShowDialog() = DialogResult.OK Then
                outputDirTextBox.Text = outputFolderBrowserDialog.SelectedPath
                goButton.Enabled = True
                outputGroupBox.Enabled = True
            End If
        End Sub

        Private Sub goButton_Click(ByVal sender As System.Object, ByVal e As System.EventArgs) Handles goButton.Click
            goButton.Enabled = False
            Me.Hide()
            'validate the folder which contains the images to be displayed
            'check if it exist
            If Directory.Exists(inputDirTextBox.Text) = True Then
                Dim fileInfo As FileInfo
                Dim dirInfo As DirectoryInfo = New DirectoryInfo(inputDirTextBox.Text)
                Dim fileInfoList As FileInfo() = dirInfo.GetFiles()
                Dim fileList As ArrayList = New ArrayList


                Dim fileExt As String
                'go through the files in the directory
                'if its a JPG or GIF file add it to the file list
                For Each fileInfo In fileInfoList
                    fileExt = LCase(fileInfo.Extension)
                    If fileExt = ".jpg" Or fileExt = ".jpeg" Or fileExt = ".gif" Then
                        fileList.Add(fileInfo.FullName)
                    End If
                Next

                'if there are one or more to display
                'start the rating process
                'otherwise display an error
                If fileList.Count >= 1 Then
                    StartRatingImages(fileList)
                Else
                    MessageBox.Show("There are no images to display!")
                End If

            Else
                MessageBox.Show("The image directory doesn't exist!")
            End If

            Me.Show()
            goButton.Enabled = True
        End Sub

        Private Sub StartRatingImages(ByVal fileList As ArrayList)
            'create a new stimuli display window
            Dim imageWindow As StimuliDisplay = New StimuliDisplay(outputDirTextBox.Text)

            'if image window setup fails display a message and return
            If Not (imageWindow.SetupStimDisplay(fileList)) Then
                imageWindow.Abort()
                imageWindow.Dispose()
                MessageBox.Show("Unable to Setup Stim Display!")
                Return
            End If

            'show the window
            imageWindow.Show()
            'start the slide show
            imageWindow.Start()
            'close the slide show
            imageWindow.Close()
            'dispose the window
            imageWindow.Dispose()
        End Sub

    End Class
End Namespace
