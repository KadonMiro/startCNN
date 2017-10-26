using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.v01.CNN
{
    class FeatureMap
    {
        public float[][,] W;
        public float[][,] dWLast;
        private int convolutionCoreSize;
        private int poolingCoreSize;
        private float summa;
        private Signal buffer;
        public Signal Output;
        public float[,] deltaPool;
        public float[][,] deltaConv;

        public FeatureMap(int convolutionCoreSize, int poolingCoreSize, int fm)//PRAVKAA
        {
            W = new float[fm][,];//PRAVKAA
            for (int i = 0; i < fm; i++)//PRAVKAA
                W[i] = new float[convolutionCoreSize, convolutionCoreSize];//PRAVKAA
            this.convolutionCoreSize = convolutionCoreSize;
            this.poolingCoreSize = poolingCoreSize;
        }

        public Signal Calculate(ref Signal[]X)
        {
            Convolution(ref X);
            Pooling();
            return Output;
        }

        public void Convolution(ref Signal[] X)
        {
            //Signal[,] buffer = new Signal[X.GetLength(0) - convolutionCoreSize + 1, X.GetLength(1) - convolutionCoreSize + 1];
            //int edgEffect = convolutionCoreSize / 2; //проверить инт

            for (int g = 0; g < X[0].X.GetLength(0) - convolutionCoreSize + 1; g++)
                for (int h = 0; h < X[0].X.GetLength(1) - convolutionCoreSize + 1; h++)
                    {
                        summa = 0;
                        for (int o = 0; o < X.Length; o++)
                            for (int i = 0; i < convolutionCoreSize; i++)
                                for (int j = 0; j < convolutionCoreSize; j++)
                                    summa += X[o].X[g + i, h + j] * W[o][i, j];

                        buffer.X[g, h] = ReLu();
                    }            
        }

        public void Pooling()
        {
            deltaPool = new float[buffer.X.GetLength(0), buffer.X.GetLength(1)]; // здаем размеры массива
            Array.Clear(deltaPool, 0, deltaPool.GetLength(0)* deltaPool.GetLength(1));

            for (int g = 0; g < buffer.X.GetLength(0); g += poolingCoreSize)
                for (int h = 0; h < buffer.X.GetLength(1); h += poolingCoreSize)
                {
                    float buf = 0;
                    int o = 0, p = 0;
                    for (int i = 0; i < poolingCoreSize; i++)
                        for (int j = 0; j < poolingCoreSize; j++)
                        {
                            if (buffer.X[i, j] > buf)
                            {
                                buf = buffer.X[i, j];
                                o = i;
                                p = j;
                            }
                        }

                    deltaPool[o + g, p + h] = 1;                
                    Output.X[g, h] = buf;
                }
        }

        public float ReLu()
        {
            return Math.Max(0, summa);
        }
    }
}
