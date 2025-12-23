/*
Copyright 2004 BIOPAC Systems, Inc.

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

Portions Copyright 2004 BIOPAC Systems, Inc.

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

namespace Biopac.API.MPDevice.Samples
{
	/// <summary>
	/// Summary description for Form1.
	/// </summary>
	public class TempControlGUI : System.Windows.Forms.Form
	{
		private System.Windows.Forms.Timer updateTempTimer;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label3;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Timer updateFanTimer;
		private System.Windows.Forms.GroupBox currentTempBox;
		private System.Windows.Forms.Label currTempLbl;
		private System.Windows.Forms.GroupBox conditionBox;
		private System.Windows.Forms.GroupBox fanStateBox;
		private System.Windows.Forms.TextBox thresholdBox;
		private System.Windows.Forms.Label fanStateLbl;
		private System.Windows.Forms.Button startStopButton;
		private System.ComponentModel.IContainer components;

		/// <summary>
		/// TecmpControlGUI constructor
		/// </summary>
		public TempControlGUI()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();

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
			this.currentTempBox = new System.Windows.Forms.GroupBox();
			this.currTempLbl = new System.Windows.Forms.Label();
			this.startStopButton = new System.Windows.Forms.Button();
			this.updateTempTimer = new System.Windows.Forms.Timer(this.components);
			this.conditionBox = new System.Windows.Forms.GroupBox();
			this.label5 = new System.Windows.Forms.Label();
			this.thresholdBox = new System.Windows.Forms.TextBox();
			this.label3 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.fanStateBox = new System.Windows.Forms.GroupBox();
			this.fanStateLbl = new System.Windows.Forms.Label();
			this.updateFanTimer = new System.Windows.Forms.Timer(this.components);
			this.currentTempBox.SuspendLayout();
			this.conditionBox.SuspendLayout();
			this.fanStateBox.SuspendLayout();
			this.SuspendLayout();
			// 
			// currentTempBox
			// 
			this.currentTempBox.Controls.Add(this.currTempLbl);
			this.currentTempBox.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.currentTempBox.Location = new System.Drawing.Point(0, 8);
			this.currentTempBox.Name = "currentTempBox";
			this.currentTempBox.Size = new System.Drawing.Size(296, 96);
			this.currentTempBox.TabIndex = 0;
			this.currentTempBox.TabStop = false;
			this.currentTempBox.Text = "Current Temperature";
			// 
			// currTempLbl
			// 
			this.currTempLbl.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom)));
			this.currTempLbl.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
			this.currTempLbl.Font = new System.Drawing.Font("Microsoft Sans Serif", 48F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.currTempLbl.ForeColor = System.Drawing.Color.Red;
			this.currTempLbl.Location = new System.Drawing.Point(8, 24);
			this.currTempLbl.Name = "currTempLbl";
			this.currTempLbl.Size = new System.Drawing.Size(280, 56);
			this.currTempLbl.TabIndex = 0;
			this.currTempLbl.Text = "OFF";
			this.currTempLbl.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
			// 
			// startStopButton
			// 
			this.startStopButton.Location = new System.Drawing.Point(240, 168);
			this.startStopButton.Name = "startStopButton";
			this.startStopButton.Size = new System.Drawing.Size(56, 24);
			this.startStopButton.TabIndex = 1;
			this.startStopButton.Text = "Start";
			this.startStopButton.Click += new System.EventHandler(this.startStopButton_Click);
			// 
			// updateTempTimer
			// 
			this.updateTempTimer.Interval = 1000;
			this.updateTempTimer.Tick += new System.EventHandler(this.updateTempTimer_Tick);
			// 
			// conditionBox
			// 
			this.conditionBox.Controls.Add(this.label5);
			this.conditionBox.Controls.Add(this.thresholdBox);
			this.conditionBox.Controls.Add(this.label3);
			this.conditionBox.Controls.Add(this.label2);
			this.conditionBox.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.conditionBox.Location = new System.Drawing.Point(0, 112);
			this.conditionBox.Name = "conditionBox";
			this.conditionBox.Size = new System.Drawing.Size(192, 80);
			this.conditionBox.TabIndex = 2;
			this.conditionBox.TabStop = false;
			this.conditionBox.Text = "Activate Fan";
			// 
			// label5
			// 
			this.label5.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label5.Location = new System.Drawing.Point(120, 24);
			this.label5.Name = "label5";
			this.label5.Size = new System.Drawing.Size(40, 16);
			this.label5.TabIndex = 6;
			this.label5.Text = "deg F,";
			// 
			// thresholdBox
			// 
			this.thresholdBox.Location = new System.Drawing.Point(64, 16);
			this.thresholdBox.Name = "thresholdBox";
			this.thresholdBox.Size = new System.Drawing.Size(48, 20);
			this.thresholdBox.TabIndex = 5;
			this.thresholdBox.Text = "80.0";
			// 
			// label3
			// 
			this.label3.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label3.Location = new System.Drawing.Point(8, 48);
			this.label3.Name = "label3";
			this.label3.Size = new System.Drawing.Size(96, 16);
			this.label3.TabIndex = 2;
			this.label3.Text = "ACTIVATE FAN";
			// 
			// label2
			// 
			this.label2.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.label2.Location = new System.Drawing.Point(8, 24);
			this.label2.Name = "label2";
			this.label2.Size = new System.Drawing.Size(64, 16);
			this.label2.TabIndex = 0;
			this.label2.Text = "IF TEMP >";
			// 
			// fanStateBox
			// 
			this.fanStateBox.Controls.Add(this.fanStateLbl);
			this.fanStateBox.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.fanStateBox.Location = new System.Drawing.Point(200, 112);
			this.fanStateBox.Name = "fanStateBox";
			this.fanStateBox.Size = new System.Drawing.Size(96, 48);
			this.fanStateBox.TabIndex = 3;
			this.fanStateBox.TabStop = false;
			this.fanStateBox.Text = "Fan State";
			// 
			// fanStateLbl
			// 
			this.fanStateLbl.Font = new System.Drawing.Font("Microsoft Sans Serif", 16F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.fanStateLbl.ForeColor = System.Drawing.Color.Red;
			this.fanStateLbl.Location = new System.Drawing.Point(8, 16);
			this.fanStateLbl.Name = "fanStateLbl";
			this.fanStateLbl.Size = new System.Drawing.Size(80, 24);
			this.fanStateLbl.TabIndex = 0;
			this.fanStateLbl.Text = "OFF";
			this.fanStateLbl.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
			// 
			// updateFanTimer
			// 
			this.updateFanTimer.Interval = 1000;
			this.updateFanTimer.Tick += new System.EventHandler(this.updateFanTimer_Tick);
			// 
			// TempControlGUI
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.ClientSize = new System.Drawing.Size(298, 199);
			this.Controls.Add(this.fanStateBox);
			this.Controls.Add(this.conditionBox);
			this.Controls.Add(this.startStopButton);
			this.Controls.Add(this.currentTempBox);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
			this.MaximizeBox = false;
			this.Name = "TempControlGUI";
			this.Text = "Temperature Control";
			this.Closing += new CancelEventHandler(Exit);
			this.currentTempBox.ResumeLayout(false);
			this.conditionBox.ResumeLayout(false);
			this.fanStateBox.ResumeLayout(false);
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			Application.Run(new TempControlGUI());
		}

		/// <summary>
		/// Handles the click event of startStopButton.
		/// Starts or stop the data acquisition system
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void startStopButton_Click(object sender, System.EventArgs e)
		{
			//case the button text is "Stop"
			if(this.startStopButton.Text == "Stop")
			{
				//stop updating GUI
				updateTempTimer.Stop();
				updateFanTimer.Stop();
				//stop acquisition system
				TempControl.StopMonitor();
				//update gui
				this.startStopButton.Text = "Start";
				this.thresholdBox.Enabled = true;
				this.currTempLbl.Text = "OFF";
				this.fanStateLbl.Text = (TempControl.Fan) ? "ON" : "OFF";
				return;
			}

			//initialize acquisition system
			//replace the string "AUTO" to the full serial number of your MP150 if
			//this program does not connect to the correct MP150
			if(!TempControl.Init("AUTO"))
			{
				//stop acquisition system
				TempControl.StopMonitor();
				//update gui
				this.startStopButton.Text = "Start";
				this.currTempLbl.Text = "ERR";
				this.fanStateLbl.Text = (TempControl.Fan) ? "ON" : "OFF";
				return;
			}

			//start acquisition system
			if(!TempControl.StartMonitor())
			{
				//stop acquisition system
				TempControl.StopMonitor();
				//update gui
				this.startStopButton.Text = "Start";
				this.currTempLbl.Text = "ERR";
				this.fanStateLbl.Text = (TempControl.Fan) ? "ON" : "OFF";
				return;
			}

			//update gui
			this.thresholdBox.Enabled = false;
			this.startStopButton.Text = "Stop";

			//start timers
			updateTempTimer.Start();
			updateFanTimer.Start();
		}

		/// <summary>
		/// Handles the tick event of updateTempTimer
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void updateTempTimer_Tick(object sender, System.EventArgs e)
		{
			updateTempTimer.Enabled = false;
			//display current temperature on GUI
			this.currTempLbl.Text = TempControl.Temperature.ToString("0.00");
			//display current temperature on LED
			TempControl.SendToLED();
			this.currTempLbl.Invalidate();
			updateTempTimer.Enabled = true;
		}

		/// <summary>
		/// Handles the tick event of updateFanTime
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void updateFanTimer_Tick(object sender, System.EventArgs e)
		{
			updateFanTimer.Enabled = false;
			//activate the fan if necessary
			TempControl.ActivateFan(System.Double.Parse(this.thresholdBox.Text));
			//update the GUI
			this.fanStateLbl.Text = (TempControl.Fan) ? "ON" : "OFF";
			updateFanTimer.Enabled = true;
		}

		/// <summary>
		/// Handle the clocing event of the application
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void Exit(object sender, CancelEventArgs e)
		{
			TempControl.StopMonitor();
		}
	}
}
