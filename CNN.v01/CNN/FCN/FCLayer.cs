using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace CNN.v01.CNN.FCN
{
    /*
     * возникли проблемы с общим массивом сигналов для всего слоя
     */
    class FCLayer
    {
        public Neuron[] neurons;

        public FCLayer(int n) // число нейронов в слое
        {
            neurons = new Neuron[n];      
        }

        public void CreateInput(int n)
        {
            for (int i = 0; i < neurons.Length; i++)
                neurons[i] = new Neuron(n);
        }

        public void Calculate()
        {
            for (int i = 0; i < neurons.Length; i++)
                neurons[i].Calculate();
        }
    }
}
