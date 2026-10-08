
using Microsoft.ML;
using Microsoft.ML.Data;

var mlcontext = new MLContext(seed: 0);

string filepath = AppContext.BaseDirectory + "cars.csv";

var fileloader = mlcontext.Data.LoadFromTextFile<CarsData>(filepath, new TextLoader.Options
{
    HasHeader = true,
    Separators = new[] { ',' },
    MissingRealsAsNaNs = true,
});

var pipeline = mlcontext.Transforms.ReplaceMissingValues(outputColumnName: "FilledAge",inputColumnName: "Age", 
    replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean)
    
    .Append(mlcontext.Transforms.ReplaceMissingValues(outputColumnName: "FilledMileage",inputColumnName: "Mileage",
    replacementMode: Microsoft.ML.Transforms.MissingValueReplacingEstimator.ReplacementMode.Mean))
    
    .Append(mlcontext.Transforms.Categorical.OneHotEncoding(outputColumnName: "BrandEncoded",inputColumnName: "Brand"))
    
    .Append(mlcontext.Transforms.Concatenate("Features", "FilledAge","FilledMileage", "BrandEncoded")).
    
    Append(mlcontext.Transforms.NormalizeMinMax("Features"));


var splits = mlcontext.Data.TrainTestSplit(fileloader, testFraction: 0.2, seed: 1);
var traindata = splits.TrainSet;
var testdata = splits.TestSet;

var transformer = pipeline.Fit(traindata);

IDataView traindataview = transformer.Transform(traindata);

IDataView testdataview = transformer.Transform(testdata);

var preview = traindataview.Preview(maxRows: 8);

foreach (var row in preview.RowView)
{
    Console.WriteLine(string.Join(" | ", row.Values.Select(v => $"{v.Key}={v.Value}")));
}

Console.WriteLine("**************************************");

var testpreivew = testdataview.Preview(maxRows: 8);

foreach (var row in testpreivew.RowView)
{
    Console.WriteLine(string.Join(" | ", row.Values.Select(v => $"{v.Key}={v.Value}")));
}

public class CarsData
{
    [LoadColumn(0)]
    public string Brand { get; set; }
    [LoadColumn(1)]
    public float Age { get; set; }
    [LoadColumn(2)]
    public float Mileage { get; set; }
    [LoadColumn(3)]
    public float Price { get; set; }
}