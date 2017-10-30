using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.v01.CNN.FCN
{
    class MiddleLayer
    {
        public Signal[] Input;
        public Signal[] Output;
        public float[] delta;

        public MiddleLayer(int n)
        {
            Input = new Signal[n];
            for (int i = 0; i < Input.Length; i++)
                Input[i] = new Signal();
            Output = new Signal[n];
            for (int i = 0; i < Output.Length; i++)
                Output[i] = new Signal();
            delta = new float[n];
        }

        public void Calculate()
        {
            for (int i = 0; i < Input.Length; i++)
                //Output[i].X = Sigmoid(Input[i].X);
                Output[i].X = Input[i].X;
        }
        public float Sigmoid(float x)
        {
            return 1 / (1 + (float)Math.Exp(-x));
        }
    }
}
