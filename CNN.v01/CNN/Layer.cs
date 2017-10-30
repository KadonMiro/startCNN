using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.v01.CNN
{
    class Layer
    {
        public Signal[] Input;
        public FeatureMap[] featureMap;

        public Layer(int featureMapSize, int sizeW, int convolutionCoreSize, int poolingCoreSize)//число карт признаков в слое
        {
            featureMap = new FeatureMap[featureMapSize];

            for (int i = 0; i < featureMap.Length; i++)
                featureMap[i] = new FeatureMap(convolutionCoreSize, poolingCoreSize, sizeW);
        }

        public void CreateInput(int size, int i, int j)
        {
            Input = new Signal[size];
            for (int k = 0; k < size; k++)
                Input[k] = new Signal(i, j);
        }

        /*
         * происходит вычисление всего слоя по всем картам признаков
         */
        public void Calculate()
        {
            for (int i = 0; i < featureMap.Length; i++)      
                featureMap[i].Calculate(ref Input);                   
        }
    }
}
