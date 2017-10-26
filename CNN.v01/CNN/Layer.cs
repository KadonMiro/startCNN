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

        //public Layer(int featureMapSize) //число карт признаков в слое 
        public Layer(int featureMapSize, int convolutionCoreSize, int poolingCoreSize)//PRAVKAA
        {
            featureMap = new FeatureMap[featureMapSize];
            for (int i = 0; i < featureMap.Length; i++)//PRAVKAA
                featureMap[i] = new FeatureMap(convolutionCoreSize, poolingCoreSize, featureMap.Length);
        }
        /*
         * происходит вычисление всего слоя по всем картам признаков
         */
        //public void Calculate(int convolutionCoreSize, int poolingCoreSize)
        public void Calculate()//PRAVKAA
        {
            for (int i = 0; i < featureMap.Length; i++)
            {
                //featureMap[i] = new FeatureMap(convolutionCoreSize, poolingCoreSize, featureMap.Length);
                featureMap[i].Calculate(ref Input);
            }                   
        }
    }
}
