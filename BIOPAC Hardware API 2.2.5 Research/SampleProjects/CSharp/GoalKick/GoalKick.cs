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
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;
using System.Data;
using System.Drawing;
using System.Drawing.Imaging;

using MP=Biopac.API.MPDevice.MPDevImports;
using MPCODE=Biopac.API.MPDevice.MPDevImports.MPRETURNCODE;
using MPTRIG=Biopac.API.MPDevice.MPDevImports.TRIGGEROPT;

namespace Biopac.API.MPDevice.Samples
{
	/// <summary>
	/// Summary description for Form1.
	/// </summary>
	public class GoalKick : System.Windows.Forms.Form
	{
		private System.ComponentModel.IContainer components;
		private System.Windows.Forms.Button startStopButton;
		private System.Windows.Forms.Timer refreshBall;
		private System.Windows.Forms.Label messageLbl;
		private System.Windows.Forms.Label ptsLabel;
		private Ball soccerball = null;
		private Point initp;
		private Point currp;
		private bool score = false;
		private int pts = 0;
		private int dx = 20;
		private int dy = -10;

		/// <summary>
		/// GoalKick constructor
		/// </summary>
		public GoalKick()
		{
			//
			// Required for Windows Form Designer support
			//
			InitializeComponent();
			this.BackgroundImage = new Bitmap(this.GetType().Assembly.GetManifestResourceStream("GoalKick.goal.bmp"));
			this.SetStyle(ControlStyles.UserPaint, true);
			this.SetStyle(ControlStyles.AllPaintingInWmPaint,true);
			this.SetStyle(ControlStyles.DoubleBuffer,true);
			initp = currp = new Point(50,350);
			soccerball = new Ball(initp);

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
			this.startStopButton = new System.Windows.Forms.Button();
			this.refreshBall = new System.Windows.Forms.Timer(this.components);
			this.messageLbl = new System.Windows.Forms.Label();
			this.ptsLabel = new System.Windows.Forms.Label();
			this.SuspendLayout();
			// 
			// startStopButton
			// 
			this.startStopButton.BackColor = System.Drawing.Color.Transparent;
			this.startStopButton.Location = new System.Drawing.Point(488, 416);
			this.startStopButton.Name = "startStopButton";
			this.startStopButton.Size = new System.Drawing.Size(104, 24);
			this.startStopButton.TabIndex = 0;
			this.startStopButton.Text = "START";
			this.startStopButton.Click += new System.EventHandler(this.startStopButton_Click);
			// 
			// refreshBall
			// 
			this.refreshBall.Tick += new System.EventHandler(this.refreshBall_Tick);
			// 
			// messageLbl
			// 
			this.messageLbl.BackColor = System.Drawing.Color.Transparent;
			this.messageLbl.Font = new System.Drawing.Font("Comic Sans MS", 36F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.messageLbl.ForeColor = System.Drawing.Color.Firebrick;
			this.messageLbl.Location = new System.Drawing.Point(0, 152);
			this.messageLbl.Name = "messageLbl";
			this.messageLbl.Size = new System.Drawing.Size(600, 40);
			this.messageLbl.TabIndex = 2;
			this.messageLbl.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
			// 
			// ptsLabel
			// 
			this.ptsLabel.BackColor = System.Drawing.Color.Transparent;
			this.ptsLabel.Font = new System.Drawing.Font("Century Gothic", 36F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((System.Byte)(0)));
			this.ptsLabel.ForeColor = System.Drawing.Color.Firebrick;
			this.ptsLabel.Location = new System.Drawing.Point(0, 0);
			this.ptsLabel.Name = "ptsLabel";
			this.ptsLabel.Size = new System.Drawing.Size(600, 40);
			this.ptsLabel.TabIndex = 1;
			this.ptsLabel.Text = "Points: 0";
			this.ptsLabel.TextAlign = System.Drawing.ContentAlignment.MiddleCenter;
			// 
			// Form1
			// 
			this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
			this.BackColor = System.Drawing.SystemColors.Control;
			this.ClientSize = new System.Drawing.Size(600, 450);
			this.Controls.Add(this.messageLbl);
			this.Controls.Add(this.ptsLabel);
			this.Controls.Add(this.startStopButton);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
			this.MaximizeBox = false;
			this.Name = "Form1";
			this.Text = "Goal Kick";
			this.Load += new System.EventHandler(this.GoalKick_Load);
			this.Closing += new CancelEventHandler(GoalKick_Closing);
			this.ResumeLayout(false);

		}
		#endregion

		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main() 
		{
			Application.Run(new GoalKick());
		}

		/// <summary>
		/// When the application loads configure the acquisition system
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>		
		private void GoalKick_Load(object sender, System.EventArgs e)
		{
			//connect to MP Device
			MPCODE rval;
			bool[] achannel = new bool[16];

			//connect to mp160
			//if the BHAPI cannot connect to the desired MP160 changed the third parameter
			//with the full serial number of the MP160
			if((rval = MP.connectMPDev(MP.MPTYPE.MP160, MP.MPCOMTYPE.MPUDP, "auto")) != MPCODE.MPSUCCESS)
			{
				abortAcquisition();
				MessageBox.Show("FAILED: connectMPDev()","MP ERROR",MessageBoxButtons.OK, MessageBoxIcon.Error);
				this.Close();
				return;
			}

			//set sample rate at 500 Hz
			if((rval = MP.setSampleRate(2.0)) != MPCODE.MPSUCCESS)
			{
				abortAcquisition();
				MessageBox.Show("FAILED: setSampleRate()","MP ERROR",MessageBoxButtons.OK, MessageBoxIcon.Error);
				this.Close();
				return;
			}
	
			//connect Tri-axial Accelerometers (TSD109F) to HLT100C
			// x -> channel 1
			// y -> channel 2
			// z -> channel 3
			
			//by default all elements of achannels are false
			//for this sample project only examine the x component of the accelerometer
			achannel[0] = true; //x component

			if((rval = MP.setAcqChannels(achannel)) != MPCODE.MPSUCCESS)
			{
				abortAcquisition();
				MessageBox.Show("FAILED: setAcqChannels()","MP ERROR",MessageBoxButtons.OK, MessageBoxIcon.Error);
				this.Close();
				return;
			}
	
			//set trigger to detect the kick setup in the on channel 1
			//Trigger at -1.5 g s positive edge
/*
			if((rval = MP.setMPTrigger(MPTRIG.MPTRIGACH,true,GsToVolts(-1.5),0)) != MPCODE.MPSUCCESS)
			{
				abortAcquisition();
				MessageBox.Show("FAILED: setMPTrigger()","MP ERROR",MessageBoxButtons.OK, MessageBoxIcon.Error);
				this.Close();
				return;
			}
*/
		}

		/// <summary>
		/// Overriden OnPaint method.
		/// </summary>
		/// <param name="e"></param>
		protected override void OnPaint(PaintEventArgs e)
		{
			if(this.refreshBall.Enabled)
			{
				this.soccerball.DrawBall(e,this.currp);
				return;
			}
					
			this.currp = this.initp;
			this.soccerball.Reset();
			this.soccerball.DrawBall(e,this.initp);

		}

		/// <summary>
		/// This function acquires 2 secs of data at 500 Hz on Channel 1
		/// it performs a simple identification of where the kick occured
		/// by finding the minimum acceleration
		/// </summary>
		/// <returns>returns the minimum accelaration (the acceleration at the moment of impact)</returns>
		private double getKickStrength()
		{
			//1000 samples = 2 sec of data
			uint samples = 1000;
			double[] data = new double[samples];
			MPCODE rval;

			//start
			//remember the trigger is set at -1.5 g 
			if((rval = MP.startAcquisition()) != MPCODE.MPSUCCESS)
			{
				MP.stopAcquisition();
				MP.disconnectMPDev();
				MessageBox.Show("FAILED: startAcquistion()","MP ERROR",MessageBoxButtons.OK, MessageBoxIcon.Error);
				return 0;
			}

			//get 2 seconds worth of data
			if((rval = MP.getMPBuffer(samples,data)) != MPCODE.MPSUCCESS)
			{
				abortAcquisition();
				MessageBox.Show("FAILED: getMPBuffer()","MP ERROR",MessageBoxButtons.OK, MessageBoxIcon.Error);
				return 0;
			}

			if((rval = MP.stopAcquisition()) != MPCODE.MPSUCCESS)
			{
				abortAcquisition();
				MessageBox.Show("FAILED: stopAcquistion()","MP ERROR",MessageBoxButtons.OK, MessageBoxIcon.Error);
				return 0;
			}


			//analyze data
			double min = System.Double.MaxValue;
			double g = 0;

			//look for the minimum acceleration
			foreach(double volts in data)
			{
				g = VoltsToGs(volts);
				min = (min > g) ? g : min;
			}

			return min;
		}
		
		/// <summary>
		/// Handles the click event
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void startStopButton_Click(object sender, System.EventArgs e)
		{	
			if(this.startStopButton.Text == "START")
			{
				double strength;

				//notify the user to Kick
				this.messageLbl.Text = "KICK!!!";
				this.messageLbl.Invalidate();
				this.startStopButton.Text = "Move your leg";
				this.Refresh();
				

				//get the strength
				strength = getKickStrength();
				this.messageLbl.Text = strength.ToString("0.000") + " G";
				this.refreshBall.Interval = 100;
				
				//if strength is positive make the ball go backwards
				if(strength > 0)
					dy = 10;
				else 
				{
					//analyze the absolute value of the kick strength
					strength = Math.Abs(strength);

					//ball goes to the left
					if(strength < 4.0)
						dy = -1 * ((int) strength + 1); //dy from -3 to -1
					//ball goes to the right
					else if(strength > 6.0)
						dy = -17-(int) (strength-6.0); //dy from -17 to -Inf
					//ball goes into the goal (-4.0 g to -6.0 g)
					else
					{ 
						//scale the strength to dy
						//map 4.0 to 1
						//map 6.0 to 11
						double smap = (5.0*strength)- 19.0;
						dy = -5 - (int) smap;  //dy from -16 to -5
					}
				}
				
				this.messageLbl.Invalidate();
				this.Refresh();

		    	this.refreshBall.Start();
				this.startStopButton.Text = "STOP";

			}
			else
			{	
				this.refreshBall.Stop();
				this.startStopButton.Text = "START";
				this.Invalidate();
				
			}
		}

		/// <summary>
		/// Checks the placement of the ball
		/// </summary>
		/// <returns>true, if the ball is in position where the animation should be stopped</returns>
		private bool evalTrajectory()
		{
			this.score = false;
			
			//check ball is in bounce
			if(currp.X >= 600 ||
			   currp.X <= 0 ||
			   currp.Y <= 0 ||
			   currp.Y >= 450+this.soccerball.Height )
				return true;

			//check altitude
			if(currp.Y < 141 || currp.Y+this.soccerball.Height > 270)
				return false;

			//check if it's in the goal
			if(currp.X > 178 && currp.X+this.soccerball.Width < 415)
			{
				this.score = true;
				return true;
			}
			else
				this.score = false;

			return true;
		}

		/// <summary>
		/// Handles the animation of the ball and if the user scores or not
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>	
		private void refreshBall_Tick(object sender, System.EventArgs e)
		{

			if(this.evalTrajectory())
			{
				this.refreshBall.Stop();
				
				if(this.score)
				{
					this.messageLbl.Text = "YOU SCORE!!!";
					this.ptsLabel.Text = "Points: " + (++pts);
					this.messageLbl.Invalidate();
				}
				else
				{
					this.messageLbl.Text = "SORRY, YOU MISSED.";
					this.messageLbl.Invalidate();
				}

				MessageBox.Show("Try Again?","GOAL KICK",MessageBoxButtons.OK, MessageBoxIcon.Question);
			
				this.startStopButton.Text = "START";
				this.messageLbl.Text = "";
				this.messageLbl.Invalidate();
				this.Invalidate();
			}
		
			//calculate new point
			this.currp.Offset(dx,dy);

			this.Invalidate();
			
		}

		/// <summary>
		/// Handles the closing even.  Calls abortAcquisition()
		/// </summary>
		/// <param name="sender"></param>
		/// <param name="e"></param>
		private void GoalKick_Closing(object sender, CancelEventArgs e)
		{
			abortAcquisition();
		}

		/// <summary>
		/// Stops the acquisition, and disconnects from the mp device
		/// </summary>
		private void abortAcquisition()
		{
			MP.stopAcquisition();
			MP.disconnectMPDev();
		}

		/// <summary>
		/// Converts Volt to G
		/// </summary>
		/// <param name="volts"></param>
		/// <returns></returns>
		private double VoltsToGs(double volts)
		{
			//based on scaling factor
			//2.897 volts maps to 1 g in the x-axis
			//2.976 volts maps to -1 g in the x-axis

			double m = -2.0/0.079;
			double c = 1.0 + (5.794/0.079);

			return (m*volts) + c;
		}

		/// <summary>
		/// Converts G to Volt
		/// </summary>
		/// <param name="gs"></param>
		/// <returns></returns>
		private double GsToVolts(double gs)
		{
			//based on scaling factor
			//2.897 volts maps to 1 g in the x-axis
			//2.976 volts maps to -1 g in the x-axis

			double m = -0.079/2.0;
			double c = 2.897 + (0.079/2.0);

			return (m*gs) + c;
		}

	}

	
}
