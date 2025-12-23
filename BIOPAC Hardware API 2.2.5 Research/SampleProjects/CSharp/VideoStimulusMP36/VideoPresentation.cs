/*
Copyright 2005-2008 BIOPAC Systems, Inc.

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

Portions Copyright 2005-2008 BIOPAC Systems, Inc.

2. Altered source versions must be plainly marked as such, and must not be 
misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
*/

using System;
using System.Management;
using System.Management.Instrumentation;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace Biopac.API.MPDevice
{
	/// <summary>
	/// Summary description for VideoDisplay.
	/// </summary>
	public class VideoPresentation : System.Windows.Forms.Form
	{
		private AxWMPLib.AxWindowsMediaPlayer MediaPlayer; //media player control
		private System.Windows.Forms.Button goButton; 
		private string url;
		private bool playDVD;
		private System.Windows.Forms.Button stopButton;
		private System.Windows.Forms.Panel dataPanel;
		private System.ComponentModel.IContainer components;
		private AcquisitionModule AcqMod;

		/// <summary>
		/// VideoPresentation constructor.
		/// </summary>
		/// <param name="file">filename of the video or the DVD driver letter</param>
		/// <param name="isDVD">true if playing a dvd video</param>
		public VideoPresentation(string file, bool isDVD )
		{
			//get rid of the warning
			components = null;
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//set double buffering parameters
			this.SetStyle(ControlStyles.UserPaint, true);
			this.SetStyle(ControlStyles.AllPaintingInWmPaint,true);
			this.SetStyle(ControlStyles.DoubleBuffer,true);

			AcqMod = new AcquisitionModule(new NewDataEventHandler(dataPanel_DrawData),1000);

			//init global variables
			url = file;
			playDVD = isDVD;
		}

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		protected override void Dispose( bool disposing )
		{
			if( disposing )
			{
				if(components != null)
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
			System.Resources.ResourceManager resources = new System.Resources.ResourceManager(typeof(VideoPresentation));
			this.MediaPlayer = new AxWMPLib.AxWindowsMediaPlayer();
			this.goButton = new System.Windows.Forms.Button();
			this.stopButton = new System.Windows.Forms.Button();
			this.dataPanel = new System.Windows.Forms.Panel();
			((System.ComponentModel.ISupportInitialize)(this.MediaPlayer)).BeginInit();
			this.SuspendLayout();
			// 
			// MediaPlayer
			// 
			this.MediaPlayer.Dock = System.Windows.Forms.DockStyle.Top;
			this.MediaPlayer.Enabled = true;
			this.MediaPlayer.Location = new System.Drawing.Point(0, 0);
			this.MediaPlayer.Name = "MediaPlayer";
			this.MediaPlayer.OcxState = ((System.Windows.Forms.AxHost.State)(resources.GetObject("MediaPlayer.OcxState")));
			this.MediaPlayer.Size = new System.Drawing.Size(394, 300);
			this.MediaPlayer.TabIndex = 0;
			// 
			// goButton
			// 
			this.goButton.Dock = System.Windows.Forms.DockStyle.Top;
			this.goButton.Location = new System.Drawing.Point(0, 300);
			this.goButton.Name = "goButton";
			this.goButton.Size = new System.Drawing.Size(394, 24);
			this.goButton.TabIndex = 1;
			this.goButton.Text = "GO";
			this.goButton.Click += new System.EventHandler(this.goButton_Click);
			// 
			// stopButton
			// 
			this.stopButton.BackColor = System.Drawing.Color.Red;
			this.stopButton.Dock = System.Windows.Forms.DockStyle.Bottom;
			this.stopButton.Enabled = false;
			this.stopButton.Font = new System.Drawing.Font("Microsoft Sans Serif", 14.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.stopButton.ForeColor = System.Drawing.Color.White;
			this.stopButton.Location = new System.Drawing.Point(0, 297);
			this.stopButton.Name = "stopButton";
			this.stopButton.Size = new System.Drawing.Size(394, 23);
			this.stopButton.TabIndex = 2;
			this.stopButton.Text = "STOP";
			this.stopButton.Visible = false;
			this.stopButton.Click += new System.EventHandler(this.stopButton_Click);
			// 
			// dataPanel
			// 
			this.dataPanel.Dock = System.Windows.Forms.DockStyle.Bottom;
			this.dataPanel.Enabled = false;
			this.dataPanel.Location = new System.Drawing.Point(0, 201);
			this.dataPanel.Name = "dataPanel";
			this.dataPanel.Size = new System.Drawing.Size(394, 96);
			this.dataPanel.TabIndex = 3;
			this.dataPanel.Visible = false;
			// 
			// VideoDisplay
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(394, 320);
			this.Controls.Add(this.goButton);
			this.Controls.Add(this.MediaPlayer);
			this.Controls.Add(this.dataPanel);
			this.Controls.Add(this.stopButton);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
			this.MaximizeBox = false;
			this.Name = "VideoDisplay";
			this.Text = "VideoDisplay";
			this.Closing += new System.ComponentModel.CancelEventHandler(this.VideoPresentation_Closing);
			((System.ComponentModel.ISupportInitialize)(this.MediaPlayer)).EndInit();
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// Initialize Media player
		/// </summary>
		/// <returns>true if the player has been properly initialized</returns>
		public bool initPlayer()
		{
			MediaPlayer.settings.autoStart = false;
			
			//validate video selection
			if(playDVD)
			{
				//check if it is a dvd
				// this code will throw an exception if the "url" is not a dvd drive letter
				// with a dvd in it
				try
				{
					WMPLib.IWMPCdrom  dvd = MediaPlayer.cdromCollection.getByDriveSpecifier(url);
					MediaPlayer.currentPlaylist.clear();
					MediaPlayer.currentPlaylist = dvd.Playlist;
				}
				catch
				{
					return false;
				}
			}
			else
			{
				//check if the file selected is a video	
				try
				{
					//add it to the medio collection
					//and set the returned media to the "media" variable
					WMPLib.IWMPMedia media = MediaPlayer.mediaCollection.add(url);
				
					//create a video media collection
					WMPLib.IWMPPlaylist videolist = MediaPlayer.mediaCollection.getByAttribute("MediaType","video");
					int count = videolist.count; 
					bool found = false;

					//check if the one of the source URL of the video list
					//matches the url of the file to be played
					for(int i=0; i < count; i++)
					{
						if(videolist.get_Item(i).sourceURL == url)
						{
							found = true;
							break;
						}
					}

					//remove the file from the media collection
					MediaPlayer.mediaCollection.remove(media,false);

					//if not found output error message
					if(!found)
					{
						MessageBox.Show("File: " + url + " is not a video file.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
						return false;
					}

					//set the current media to the media with the url
					MediaPlayer.currentMedia = media;
				}
				catch
				{
					MessageBox.Show("File: " + url + " is not a video file.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
					return false;
				}
			}
		
			return true;
		}

		/// <summary>
		/// Handles the event when the go button is clicked
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void goButton_Click(object sender, System.EventArgs e)
		{
			Enabled = Visible = false;

			//initialize acquisition module
			if(!AcqMod.initModule())
			{
				MessageBox.Show("Acquisition module failed to initialize.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
				Close();
				return;
			}

			//change the window to fullscreen mode
		    switchToFullScreen();
			//play the media
			MediaPlayer.Ctlcontrols.play();
			
			Enabled = Visible = true;
		}

		/// <summary>
		/// Changes the layout of the window to fullscreen
		/// </summary>
		private void switchToFullScreen()
		{
			//Make the form fullscreen
			//form
			this.Left = 0;
			this.Top = 0;
			this.WindowState = FormWindowState.Maximized;
			this.BackColor = Color.Black;
			this.FormBorderStyle = FormBorderStyle.None;
			this.Invalidate();

			//Make the media player take 80% of the height of the screen
			//player
			MediaPlayer.uiMode = "None";
			MediaPlayer.stretchToFit = true;
			MediaPlayer.Height = (int) (Screen.PrimaryScreen.Bounds.Height * .80);
			MediaPlayer.Invalidate();
		
			//Make the data visualization aread 20% of the height of the screen - the height of the stop button
			//dataPanel
			dataPanel.Height = (int) (Screen.PrimaryScreen.Bounds.Height * .20) - stopButton.Height + 2;
			dataPanel.Width = (int) (Screen.PrimaryScreen.Bounds.Width);
			dataPanel.Enabled = dataPanel.Visible = true;
			dataPanel.BackColor = Color.Black;

			//show the stop button
			//stopButton
			stopButton.Enabled = stopButton.Visible = true;

			//hide the go button
			//goButton
			goButton.Enabled = goButton.Visible = false;

		}

		/// <summary>
		/// Handles the event when the stop button is clicked
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void stopButton_Click(object sender, System.EventArgs e)
		{
			stopButton.Enabled = false;
			AcqMod.StopAcquisition();
			//add save stuff here
			Close();
		}

		/// <summary>
		/// Handles the event when the form is being closed
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void VideoPresentation_Closing(object sender, System.ComponentModel.CancelEventArgs e)
		{
			//stop the video playback
			MediaPlayer.Ctlcontrols.stop();
			//stop the acquisition
			AcqMod.StopAcquisition();
		}

		/// <summary>
		/// Handles the NewData event from the acquisition module
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void dataPanel_DrawData(object sender, NewDataEventArgs e)
		{
			//create graphics
			Graphics g = this.dataPanel.CreateGraphics();
			//draw the acquisition data on the data panel
			AcqMod.DrawData(g);
		}
	}
}
