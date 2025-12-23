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
Imports System.Collections
Imports System.IO
Imports System.Xml
Imports System.Xml.Serialization


Namespace Biopac.API.MPDevice
    'structure for holding image stimulus information
    Public Structure ImageStimulus
        Public Score As Integer 'numeric score
        Public Location As String 'file location
        Public UniqueID As String 'unique id of the image
        <XmlIgnoreAttribute()> _
        Public Waveform As ArrayList 'physiological response waveform
        <XmlIgnoreAttribute()> _
        Public theImage As Bitmap 'the actual image
        <XmlIgnoreAttribute()> _
        Public theWaveImage As Bitmap 'the image representation of the Waveform
    End Structure

    Public Class StimulusReport

        <XmlArrayItem("Stimulus")> _
        Public StimulusList() As ImageStimulus
        <XmlIgnoreAttribute()> _
        Private OutputDir As String


        Public Sub New()
            'needed for serialization
            'intentionally left empty
        End Sub

        Public Sub New(ByVal numImages As Integer, ByVal dir As String)
            ReDim StimulusList(numImages) 'resize the array
            OutputDir = dir 'the output directory
        End Sub

        Public Sub GenerateReport()
            WriteWaveformFiles()
            WriteXSLFile()
            WriteXMLFile()
        End Sub

        Private Sub WriteWaveformFiles()
            Dim count As Integer = StimulusList.Length
            Dim c As Integer

            For c = 0 To count - 1
                GenerateWaveformImageToDisk(c)
            Next
        End Sub

        Private Sub WriteXSLFile()
            Dim xslFileStream As New FileStream(OutputDir + "\stimreport.xsl", FileMode.Create)
            'get the xml stylesheet.  it's compiled as a resource
            Dim xslTextReader As TextReader = New StreamReader(Me.GetType().Assembly.GetManifestResourceStream("ImageStim.stimreport.xsl"))
            'create a new text file from the xslFileStream
            Dim xslTextWriter As TextWriter = New StreamWriter(xslFileStream)

            'write the xml stylesheet from memory to disk
            xslTextWriter.Write(xslTextReader.ReadToEnd())

            'close the file streams
            xslTextWriter.Close()
            xslTextReader.Close()
            xslFileStream.Close()

        End Sub

        Private Sub GenerateWaveformImageToDisk(ByVal index As Integer)

            Dim waveform As ArrayList = StimulusList(index).Waveform

            'find the max and min of the waveform
            Dim maxVolt As Single = System.Single.MinValue
            Dim minVolt As Single = System.Single.MaxValue

            Dim val As Single
            Dim s As Integer
            For s = 0 To waveform.Count - 1
                val = Convert.ToSingle(waveform(s))

                If val > maxVolt Then
                    maxVolt = val
                End If

                If val < minVolt Then
                    minVolt = val
                End If

            Next

            'plot the waveform onto the image
            Dim waveformImg As Bitmap = StimulusList(index).theWaveImage
            Dim waveformCoord(waveform.Count - 1) As PointF

            Dim pixelWidth As Single = waveformImg.Width
            Dim pixelHeight As Single = waveformImg.Height
            Dim pixelInterval As Single = pixelWidth / waveform.Count

            'for convertion volts to pixels (linear convertion)
            Dim maxPixelHeight As Single = pixelHeight * 0.1 'allow 10% off the top
            Dim minPixelHeight As Single = pixelHeight * 0.9 'allow 10% off the bottom
            Dim slope As Single = (maxPixelHeight - minPixelHeight) / (maxVolt - minVolt)
            Dim offset As Single = maxPixelHeight - (slope * maxVolt)

            Dim x As Single
            Dim y As Single

            Dim t As Single = 0
            'convert waveform to data points
            For s = 0 To waveform.Count - 1
                x = t
                y = (slope * Convert.ToSingle(waveform(s))) + offset
                waveformCoord(s) = New PointF(x, y)
                t = t + pixelInterval
            Next

            'get the graphic object of the image
            Dim g As Graphics = Graphics.FromImage(waveformImg)
            'conofigure thre graphics object
            g.CompositingQuality = Drawing2D.CompositingQuality.HighQuality
            g.InterpolationMode = Drawing2D.InterpolationMode.HighQualityBicubic

            g.Clear(Color.White)

            Dim redPen As Pen = New Pen(Color.Red, 2.0)

            'draw the image
            g.DrawLines(redPen, waveformCoord)

            'save the image as a jpeg to disk
            waveformImg.Save(OutputDir + "\" + StimulusList(index).UniqueID + ".jpeg", Imaging.ImageFormat.Jpeg)

        End Sub

        Private Sub WriteXMLFile()
            'Serialize this Stimulus Report Object to a XML file
            Dim serializer As New XmlSerializer(GetType(StimulusReport))
            Dim fileStream As New FileStream(OutputDir + "\StimulusReport-" + System.DateTime.Now.Ticks.ToString() + ".xml", FileMode.Create)
            'create an xml writer from the filestream
            Dim writer As New XmlTextWriter(fileStream, System.Text.Encoding.UTF8)

            'configure xml writer
            writer.Formatting = Formatting.Indented
            writer.WriteProcessingInstruction("xml", "version=""1.0"" encoding=""utf-8""")
            'remember to use the stylesheet so that the xml will render in the browser formatted
            writer.WriteProcessingInstruction("xml-stylesheet", "type=""text/xsl"" href=""stimreport.xsl""")

            'serialize this object to the xml writer
            serializer.Serialize(writer, Me)

            'close streams
            writer.Close()
            fileStream.Close()
        End Sub
    End Class
End Namespace