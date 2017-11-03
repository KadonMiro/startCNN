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
using System.Diagnostics;// для таймера

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
            int[] deepConvolutionLayers = {3, 5};
            int[] convolutionCoreSize = {3, 3};
            int[] poolingCoreSize = {2, 2};
            int fullConnectedLaersSize = 1;
            ///////////////////////////
            
            float[][] trueOutput = new float[2][];
            trueOutput[0] = new float[] { 1, 0};
            trueOutput[1] = new float[] { 0, 1};

            Bitmap X = CreateBitmapX();
            Bitmap O = CreateBitmapO();
            int i = 0;

            Stopwatch sw = new Stopwatch();
            sw.Start();

            Net net = new Net(ref deepConvolutionLayers, ref convolutionCoreSize, ref poolingCoreSize, fullConnectedLaersSize,ref FirstStart);

            for (int j = 0; j < 40; j++)
            {
                float error = 0;

                    while (i < 2)
                    {
                        signal = BitmapToByteRgb(O);
                        //if (i == 0) signal = BitmapToByteRgb(X);
                        //else signal = BitmapToByteRgb(O);

                        float []temp1 = net.Calculate(signal);
                        for (int o = 0; o < temp1.Length; o++)
                            richTextBox1.Text += temp1[o].ToString() + "   ";
                        richTextBox1.Text += '\n';
                        //error += net.DeepLerning(trueOutput[0], 0.1f, 0.05f);
                        richTextBox1.Text += net.DeepLerning(trueOutput[1], 0.01f, 0.005f).ToString() + "\n";
                        i++;
                    }
                i = 0;
                    //error /= 9;
                    //richTextBox1.Text += error.ToString() + "\n";
                }
            
            sw.Stop();
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
            BitmapData bd = bmp.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly,PixelFormat.Format24bppRgb);
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

        public Bitmap CreateBitmapX()
        {
            Bitmap bmp = new Bitmap(4, 4);
            for (int i = 0; i < 4; i++)
                {
                    bmp.SetPixel(i, i, Color.FromArgb(255, 255, 255));
                    bmp.SetPixel(3 - i, i, Color.FromArgb(255, 255, 255));
                }
            return bmp;
        }

        public Bitmap CreateBitmapO()
        {
            Bitmap bmp = new Bitmap(4, 4);
            for (int i = 0; i < 4; i++)
            {
                bmp.SetPixel(i, 0, Color.FromArgb(255, 255, 255));
                bmp.SetPixel(0, i, Color.FromArgb(255, 255, 255));
                bmp.SetPixel(3, i, Color.FromArgb(255, 255, 255));
                bmp.SetPixel(i, 3, Color.FromArgb(255, 255, 255));
            }
            return bmp;
        }

        private void richTextBox1_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
