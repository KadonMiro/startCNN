﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNN.v01.CNN
{
    class Signal
    {
        public float[,] X;

        public Signal(int i, int j)
        {
            X = new float[i, j];
        }
    }
}
