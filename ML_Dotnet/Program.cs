
using Microsoft.ML;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.Linq;

var mlContext = new MLContext(seed: 0);
var random = new Random(15);

var sales = Enumerable.Range(0, 48)
  .Select(month => new MonthlySales
  {
      Sales = (float)(
      1000 + month * 15
      + 200 * Math.Sin(2 * Math.PI * month / 12)
      + random.Next(-30, 30))
  }).ToList();


var data = mlContext.Data.LoadFromEnumerable<MonthlySales>(sales);


public class MonthlySales { 
    public float Sales { get; set; }
}