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
            int[] deepConvolutionLayers = {3, 3, 4};
            int[] convolutionCoreSize = {5, 3, 3};
            int[] poolingCoreSize = { 2, 2, 2};
            int fullConnectedLaersSize = 3;
            ///////////////////////////
            
            float[][] trueOutput = new float[3][];
            trueOutput[0] = new float[] { 1, 0, 0 };
            trueOutput[1] = new float[] { 0, 1, 0 };
            trueOutput[2] = new float[] { 0, 0, 1 };

            string filename = @"C:\Users\Женя\Documents\GitHub\startCNN\test.txt";
            int i = 9;

            Stopwatch sw = new Stopwatch();
            sw.Start();

            Net net = new Net(ref deepConvolutionLayers, ref convolutionCoreSize, ref poolingCoreSize, fullConnectedLaersSize,ref FirstStart);

            for (int j = 0; j < 100; j++)
            {
                float error = 0;
                using (StreamReader g = new StreamReader(filename, System.Text.Encoding.Default))
                {
                    int k = 0;
                    while (i < 10)
                    {
                        if (!g.EndOfStream)
                        {
                            if (k == 3) k = 0;
                            string file = @g.ReadLine();
                            Bitmap test = LoadBitmap(file);
                            signal = BitmapToByteRgb(test);

                            float []temp1 = net.Calculate(signal);
                            for (int o = 0; o < temp1.Length; o++)
                                //richTextBox1.Text += temp1[o].ToString() + "   ";
                            //richTextBox1.Text += '\n';
                            error += net.DeepLerning(trueOutput[k], 0.1f, 0.05f);
                            //richTextBox1.Text += net.DeepLerning(trueOutput[k], 0.1f, 0.0f).ToString() + "\n";
                            k++;
                        }
                        else break;
                    }
                    error /= 9;
                    richTextBox1.Text += error.ToString() + "\n";
                    g.Close();
                }
            }
            sw.Stop();
            // string time = string.Format("time {0}", sw.ElapsedMilliseconds / 100.0);

            richTextBox1.Text += "learn end" + "\n";

            string temp = @"C:\Users\Женя\Documents\GitHub\startCNN\mushroom.png";
            Bitmap buf = LoadBitmap(temp);
            signal = BitmapToByteRgb(buf);

            float[] res = net.Calculate(signal);
            for (int j = 0; j < res.Length; j++)
                richTextBox1.Text += res[j].ToString() + "   ";

            richTextBox1.Text += "\n";
            temp = @"C:\Users\Женя\Documents\GitHub\startCNN\plus.png";
            buf = LoadBitmap(temp);
            signal = BitmapToByteRgb(buf);

            res = net.Calculate(signal);
            for (int j = 0; j < res.Length; j++)
                richTextBox1.Text += res[j].ToString() + "   ";

            richTextBox1.Text += "\n";
            temp = @"C:\Users\Женя\Documents\GitHub\startCNN\sharingan.png";
            buf = LoadBitmap(temp);
            signal = BitmapToByteRgb(buf);

            res = net.Calculate(signal);
            for (int j = 0; j < res.Length; j++)
                richTextBox1.Text += res[j].ToString() + "   ";
            richTextBox1.Text += "\n";
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

        private void richTextBox1_TextChanged(object sender, EventArgs e)
        {
         
        }
    }
}
