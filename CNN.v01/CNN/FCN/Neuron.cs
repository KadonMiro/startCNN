using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.v01.CNN.FCN
{
    /*
     * на самом деле здесь что-то не чисто и имеются недопонимания.
     * как один набор сигналов Х распределяется на все объекты этого класса в слое?
     */
    class Neuron
    {
        public Signal[] X;
        public float[] W;
        public float[] dWLast;
        private float summa;
        public Signal Output;
        public float delta;

        public Neuron(int link)
        {
            X = new Signal[link];
            for (int i = 0; i < X.Length; i++)
                X[i] = new Signal();
            W = new float[link];
            dWLast = new float[link];
            Output = new Signal();
        }
        public void Calculate()
        {
            summa = 0;
            for (int i = 0; i < X.Length; i++)
                summa += X[i].X * W[i];
            Output.X = Sigmoid();
        }

        public float Sigmoid()
        {
            return 1 / (1 + (float)Math.Exp(-summa));
        }
    }
}
