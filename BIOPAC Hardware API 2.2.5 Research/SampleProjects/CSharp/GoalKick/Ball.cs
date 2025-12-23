/*
Copyright 2004-2006 BIOPAC Systems, Inc

This software is provided 'as-is', without any express or implied warranty. In no 
event will the authors be held liable for any damages arising from the use of this 
software.

Permission is granted to anyone to use this software for any purpose, including 
commercial applications, and to alter it and redistribute it freely, subject to the 
following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that 
you wrote the original software. If you use this software in a product, an 
acknowledgment (see the following) in the product documentation is required.

Portions Copyright 2004-2006 BIOPAC Systems, Inc

2. Altered source versions must be plainly marked as such, and must not be 
misrepresented as being the original software.

3. This notice may not be removed or altered from any source distribution.
*/

using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Forms;

namespace Biopac.API.MPDevice.Samples
{
	/// <summary>
	/// Summary description for Ball.
	/// </summary>
	public class Ball
	{
		private Bitmap initimage;
		private Bitmap currimage;
		private Point oPoint;
		private Random r;

		#region Properties
		/// <summary>
		/// Width of the ball
		/// </summary>
		public int Width
		{
			get
			{
				return this.currimage.Width;
			}

		}

		/// <summary>
		/// Height of the ball
		/// </summary>
		public int Height
		{
			get
			{
				return this.currimage.Height;
			}
		}

		/// <summary>
		/// Gets the next image
		/// </summary>
		public Bitmap nextImage
		{
			get
			{
				switch(r.Next(10000) % 10)
				{
					case 0: 
						currimage.RotateFlip(RotateFlipType.Rotate90FlipNone);
						break;
					case 1:
						currimage.RotateFlip(RotateFlipType.Rotate180FlipNone);
						break;
					case 3:
						currimage.RotateFlip(RotateFlipType.Rotate270FlipNone);
						break;
					case 4:
						currimage.RotateFlip(RotateFlipType.RotateNoneFlipXY);
						break;
					case 5:
						currimage.RotateFlip(RotateFlipType.Rotate90FlipY);
						break;
					default:
						currimage.RotateFlip(RotateFlipType.Rotate270FlipXY);
						break;
				}
				
				return currimage;
			}
		}
		#endregion

		#region Public methods
		/// <summary>
		/// Ball Constructor
		/// </summary>
		/// <param name="p">the orginal starting poing</param>
		public Ball(Point p)
		{
			//get image from resource stream
			initimage = currimage = new Bitmap(this.GetType().Assembly.GetManifestResourceStream("GoalKick.ball.gif"));
			r = new Random();
			oPoint = p;
		}

		/// <summary>
		/// Resets the ball
		/// </summary>
		public void Reset()
		{
			this.currimage = initimage;
		}

		/// <summary>
		/// Draw the ball
		/// </summary>
		/// <param name="e">Paint Event Args</param>
		/// <param name="p">Point to draw image</param>
		public void DrawBall(PaintEventArgs e, Point p)
		{
			//calculate scaling factor
			double scale = ((.2 - 1)/Math.Abs(150 - oPoint.Y)) * Math.Abs(p.Y - oPoint.Y) + 1;
			//scale width and height
			int width = (int) (this.currimage.Width * scale);
			int height = (int) (this.currimage.Height * scale);
			
			//20 pixels minimum width
			if(width < 20)
				width = 20;

			//20 pixels minimum height
			if(height < 20)
				height = 20;
		
			this.currimage = new Bitmap(this.currimage, new Size(width,height));
			e.Graphics.DrawImage(this.nextImage, p);
		}
		#endregion

	}
}
