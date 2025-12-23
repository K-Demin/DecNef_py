/*
Copyright 2005-2006 BIOPAC Systems, Inc.

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

Portions Copyright 2005-2006 BIOPAC Systems, Inc.

2. Altered source versions must be plainly marked as such, and must not be 
misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
*/

using System;
using System.IO;
using System.Management;
using System.Management.Instrumentation;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;

namespace Biopac.API.MPDevice
{
	/// <summary>
	/// Summary description for VideoStimulusGUI.
	/// </summary>
	public class VideoStimulusGUI : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Button dvdButton;
		private System.Windows.Forms.Button regVideo;
		private System.Windows.Forms.Label orLabel;
		private System.Windows.Forms.OpenFileDialog mediaFileDialog;
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.Container components = null;

		public VideoStimulusGUI()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();
			
		
			//
			// TODO: Add any constructor code after InitializeComponent call
			//
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if (components != null) 
				{
					components.Dispose();
				}
			}
			base.Dispose( disposing );
		}

		#region Windows Form Designer generated code
		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.dvdButton = new System.Windows.Forms.Button();
			this.regVideo = new System.Windows.Forms.Button();
			this.orLabel = new System.Windows.Forms.Label();
			this.mediaFileDialog = new System.Windows.Forms.OpenFileDialog();
			this.SuspendLayout();
			// 
			// dvdButton
			// 
			this.dvdButton.Location = new System.Drawing.Point(8, 8);
			this.dvdButton.Name = "dvdButton";
			this.dvdButton.Size = new System.Drawing.Size(200, 24);
			this.dvdButton.TabIndex = 0;
			this.dvdButton.Text = "Play DVD";
			this.dvdButton.Click += new System.EventHandler(this.dvdButton_Click);
			// 
			// regVideo
			// 
			this.regVideo.Location = new System.Drawing.Point(8, 72);
			this.regVideo.Name = "regVideo";
			this.regVideo.Size = new System.Drawing.Size(200, 24);
			this.regVideo.TabIndex = 3;
			this.regVideo.Text = "Play Video";
			this.regVideo.Click += new System.EventHandler(this.regVideo_Click);
			// 
			// orLabel
			// 
			this.orLabel.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.orLabel.Location = new System.Drawing.Point(8, 40);
			this.orLabel.Name = "orLabel";
			this.orLabel.Size = new System.Drawing.Size(200, 24);
			this.orLabel.TabIndex = 2;
			this.orLabel.Text = "OR";
			this.orLabel.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
			// 
			// mediaFileDialog
			// 
			this.mediaFileDialog.Filter = "All files (*.*)|*.*";
			this.mediaFileDialog.Title = "Select Video File";
			// 
			// VideoStimulusGUI
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(216, 102);
			this.Controls.Add(this.orLabel);
			this.Controls.Add(this.regVideo);
			this.Controls.Add(this.dvdButton);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
			this.MaximizeBox = false;
			this.Name = "VideoStimulusGUI";
			this.Text = "Video Stimulus";
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			Application.Run(new VideoStimulusGUI());
		}

		private void dvdButton_Click(object sender, System.EventArgs e)
		{
			Enabled = Visible = false;

			//check if there is a DVD on one of the drives
			//by querying for drives that contain a CD with UDF format.
			//DVD's are formatted UDF
			ManagementObjectSearcher searcher =	new ManagementObjectSearcher("Select * from Win32_LogicalDisk Where FileSystem = 'UDF'");
			string dvdDriveID = null;
			
			foreach (ManagementObject udfDisc in searcher.Get()) 
			{
				//check if it's a DVD movie disk
				//DVD movies have a folder called "VIDEO_TS"
				if( Directory.Exists(udfDisc.GetPropertyValue("DeviceID").ToString()+"\\VIDEO_TS") )
				{
					dvdDriveID = udfDisc.GetPropertyValue("DeviceID").ToString();
					break;
				}
			}

			if(dvdDriveID == null)
			{
				MessageBox.Show("Unable to find DVD!", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
			}
			else
			{
				//create video presentation window
				VideoPresentation vidSetup = new VideoPresentation(dvdDriveID, true);
				
				//if initialization is successful show the window
				if(vidSetup.initPlayer())
					vidSetup.ShowDialog();
			}

			Enabled = Visible = true;
		}

		private void regVideo_Click(object sender, System.EventArgs e)
		{
			Enabled = Visible = false;

			if(mediaFileDialog.ShowDialog() == DialogResult.OK)
			{
				//create video presentation window
				VideoPresentation vidSetup = new VideoPresentation(mediaFileDialog.FileName, false);

				//if initialization is successful show the window
				if(vidSetup.initPlayer())
					vidSetup.ShowDialog();
			}

			Enabled = Visible = true;
		}
	}
}
