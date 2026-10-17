
using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.TimeSeries;
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

var history = sales.Take(42).ToList();

var actual = sales.Skip(42).Select(x=>x.Sales).ToList();



var data = mlContext.Data.LoadFromEnumerable<MonthlySales>(history);

var pipeline = mlContext.Forecasting.ForecastBySsa(
    outputColumnName: nameof(ForcastedSales.ForcastedSale),
    inputColumnName: nameof(MonthlySales.Sales), 
    windowSize: 12, 
    seriesLength: 36, 
    trainSize: history.Count,
    horizon: 6, 
    confidenceLevel: 0.95f, 
    confidenceLowerBoundColumn: nameof(ForcastedSales.LowerBound),
    confidenceUpperBoundColumn: nameof(ForcastedSales.UpperBound));

var model = pipeline.Fit(data);


var engine = model.CreateTimeSeriesEngine<MonthlySales, ForcastedSales>(mlContext);

var predict = engine.Predict();

for(int i = 0;  i < 6; i++)
{
    Console.WriteLine($"Month: {i + 1},   Forcast Range = {predict.LowerBound[i]:F0}-{predict.UpperBound[i]:F0} , Actual = {actual[i]:F0}");
}

var miss = actual.Zip(predict.ForcastedSale, (a, f) => Math.Abs(a - f)).Average();

Console.WriteLine($"Average Miss: {miss}");

public class MonthlySales { 
    public float Sales { get; set; }
}

public class ForcastedSales
{
    public float[] ForcastedSale { get; set; }
    public float[] LowerBound { get; set; }
    public float[] UpperBound   { get; set; }
}
