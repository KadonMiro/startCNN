using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Drawing.Imaging;

using CNN.v01.CNN;

namespace CNN.v01
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            bool FirstStart = true;
            byte[,,] signal; // бесит
            int[] deepConvolutionLayers = {3, 5, 5, 7, 5, 4};
            int[] convolutionCoreSize = {3, 3, 3, 3, 3, 3};
            int[] poolingCoreSize = {2, 2, 2, 2, 2, 2};
            int fullConnectedLaersSize = 2;
            //int X = 32, Y = 32; // считаю, что это размеры исходного изображения
            string filename = @"C:\Users\Женя\Documents\Visual Studio 2015\Projects\CNN.v01\test.jpg";

            Bitmap test = LoadBitmap(filename);
            signal = BitmapToByteRgb(test);

            Net net = new Net(ref deepConvolutionLayers, ref convolutionCoreSize, ref poolingCoreSize, fullConnectedLaersSize);
            net.Calculate(signal, ref FirstStart);

        }

        public static Bitmap LoadBitmap(string fileName)
        {
            using (FileStream fs = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read))
                return new Bitmap(fs);
        }

        public unsafe static byte[,,] BitmapToByteRgb(Bitmap bmp)
        {
            int width = bmp.Width,
                height = bmp.Height;
            byte[,,] res = new byte[3, height, width];
            BitmapData bd = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);
            try
            {
                byte* curpos;
                for (int h = 0; h < height; h++)
                {
                    curpos = ((byte*)bd.Scan0) + h * bd.Stride;
                    for (int w = 0; w < width; w++)
                    {
                        res[2, h, w] = *(curpos++);
                        res[1, h, w] = *(curpos++);
                        res[0, h, w] = *(curpos++);
                    }
                }
            }
            finally
            {
                bmp.UnlockBits(bd);
            }
            return res;
        }

        private void richTextBox1_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
