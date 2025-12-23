/*
Copyright 2004-2023 BIOPAC Systems, Inc.

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

Portions Copyright 2004-2023 BIOPAC Systems, Inc.

2. Altered source versions must be plainly marked as such, and must not be 
misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
*/

using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;

using MP=Biopac.API.MPDevice.MPDevImports;
using MPCODE=Biopac.API.MPDevice.MPDevImports.MPRETURNCODE;
using TRIGOPT=Biopac.API.MPDevice.MPDevImports.TRIGGEROPT;
using MPTYPE=Biopac.API.MPDevice.MPDevImports.MPTYPE;
using MPCOM=Biopac.API.MPDevice.MPDevImports.MPCOMTYPE;

namespace Biopac.API.MPDevice.Samples
{
	/// <summary>
	/// Summary description for Form1.
	/// </summary>
	public class Biofeedback : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Label instructMessage;
		private System.Windows.Forms.Timer updateTimer;
		private System.ComponentModel.IContainer components;

		/// <summary>
		/// Biofeedback constructor
		/// </summary>
		public Biofeedback()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

			//initialize acquisition system
			if(initAcquistionUnit())
			{
				this.instructMessage.Text = "Keep the RED circle inside the GREEN circle.";
				this.updateTimer.Start();
			}
			else
			{
				this.instructMessage.Text = "Failed to connect.  Exit the Application and try again.";
			}
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
			this.components = new System.ComponentModel.Container();
			this.instructMessage = new System.Windows.Forms.Label();
			this.updateTimer = new System.Windows.Forms.Timer(this.components);
			this.SuspendLayout();
			// 
			// instructMessage
			// 
			this.instructMessage.Dock = System.Windows.Forms.DockStyle.Top;
			this.instructMessage.Font = new System.Drawing.Font("Arial Black", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.instructMessage.ForeColor = System.Drawing.Color.DarkBlue;
			this.instructMessage.Location = new System.Drawing.Point(0, 0);
			this.instructMessage.Name = "instructMessage";
			this.instructMessage.Size = new System.Drawing.Size(292, 56);
			this.instructMessage.TabIndex = 0;
			this.instructMessage.Text = "Connecting....";
			this.instructMessage.TextAlign = System.Drawing.ContentAlignment.TopCenter;
			// 
			// updateTimer
			// 
			this.updateTimer.Interval = 500;
			this.updateTimer.Tick += new System.EventHandler(this.UpdateScreen);
			// 
			// Biofeedback
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(292, 266);
			this.Controls.Add(this.instructMessage);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.Name = "Biofeedback";
			this.Text = "Biofeedback";
			this.WindowState = System.Windows.Forms.FormWindowState.Maximized;
			this.Closing += new System.ComponentModel.CancelEventHandler(this.Biofeedback_Closing);
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			Application.Run(new Biofeedback());
		}

		
		/// <summary>
		/// Method that initializes and starts the acquisition system
		/// Acquires at 500Hz on channel 1
		/// </summary>
		/// <returns>true, if the MP unit has been configured and acquiring data. false, otherwise</returns>
		private bool initAcquistionUnit()
		{
			//connect to the MP device
			MPCODE retval = MPCODE.MPSUCCESS;
			int i = 0;
			int retry = 3;

			do
			{
				//remember to change the parameters to suit your MP configuration
				//Auto connect to MP160 was introduced in BHAPI 2.2.1
				//passing "AUTO" or "auto" instead of the full serial number of the MP160
				//will cause BHAPI to connect to the first respoding MP160.  This is usually
				//the closest MP160 to the host computer.
				retval = MP.connectMPDev(MPTYPE.MP160,MPCOM.MPUDP,"AUTO");
			}
			while((retval != MPCODE.MPSUCCESS) && ++i < retry);

			//error
			if(retval != MPCODE.MPSUCCESS)
				return false;

			//set sample rate to 500 Hz
			//2 msec per sample = 500 Hz
			retval = MP.setSampleRate(2);

			//error
			if(retval != MPCODE.MPSUCCESS)
				return false;

			//acquire on channel 1
			bool[] aCH = {true, false, false, false,
						  false, false, false, false,
						  false, false, false, false,
						  false, false, false, false};

			retval = MP.setAcqChannels(aCH);
			
			//error
			if(retval != MPCODE.MPSUCCESS)
				return false;

			//start acquisition
			retval = MP.startAcquisition();

			//error
			if(retval != MPCODE.MPSUCCESS)
				return false;

			return true;
		}

		/// <summary>
		/// Handles the closing event.  Calls abortAcquisition()
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void Biofeedback_Closing(object sender, CancelEventArgs e)
		{
			abortAcquisition();
		}
		
		/// <summary>
		/// Stops the timer, stops the acquisition, and disconnects from the mp device
		/// </summary>
		private void abortAcquisition()
		{
			this.instructMessage.Text = "Communication Error.  Exit the Application and try again.";
			this.updateTimer.Stop();
			MP.stopAcquisition();
			MP.disconnectMPDev();
		}

		/// <summary>
		/// Updates the size of the Red and Green circles
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void UpdateScreen(object sender, EventArgs e)
		{
			this.updateTimer.Enabled = false;
			
			SolidBrush brush = null;
			Pen pen = null;
			Graphics g = this.CreateGraphics();
			float actualDiameter;
			float targetDiameter = 200.0F;
			float x;
			float y;

			//clear the canvas
			g.Clear(this.BackColor);
			
			//draw target circle in green
			brush = new SolidBrush(Color.Green);
			x = (this.Size.Width/2) - (targetDiameter/2);
			y = (this.Size.Height/2) - (targetDiameter/2);
			g.FillEllipse(brush, x, y, targetDiameter, targetDiameter);

			//draw the actual circle
			if(!GetCircleDiameter(out actualDiameter))
			{
				abortAcquisition();
				return;
			}

			brush = new SolidBrush(Color.Red);
			x = (this.Size.Width/2) - (actualDiameter/2);
			y = (this.Size.Height/2) - (actualDiameter/2);
			g.FillEllipse(brush, x, y, actualDiameter, actualDiameter);

			//draw target circle outline in green
			pen = new Pen(Color.Green,2.0F);
			x = (this.Size.Width/2) - (targetDiameter/2);
			y = (this.Size.Height/2) - (targetDiameter/2);
			g.DrawEllipse(pen,x,y,targetDiameter,targetDiameter);

			this.updateTimer.Enabled = true;
		}

		/// <summary>
		/// Polls the MP unit for its current sample
		/// </summary>
		/// <param name="d">d, out paramter where the value for channel 1 is stored</param>
		/// <returns>false, if a communication error occurs</returns>
		private bool GetCircleDiameter(out float d)
		{
			double[] samples = new double[16];
			d = 0;
			
			if(MP.getMostRecentSample(samples) != MPCODE.MPSUCCESS)
				return false;

			// 0 volts maps to 50 pixels
			// 10 volts maps to 600 pixels
			// linear scaling
			d = (float) ((75.0*samples[0]) + 50.0F);
			
			return true;
		}
	}
}
