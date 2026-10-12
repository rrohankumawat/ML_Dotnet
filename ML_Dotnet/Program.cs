using Microsoft.ML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 0);

var random = new Random(10);
List<Customer> customers = new();

void GenerateCustomers(int count , float income, float spending)
{
    for(int i = 0; i < count; i++)
    {
        customers.Add(new Customer
        {
            AnualIncomeK = income + random.Next(-8, 8),
            SpendingScore = spending + random.Next(-8, 8)
        });
    }
}

GenerateCustomers(200, 30, 70);
GenerateCustomers(200, 85, 65);
GenerateCustomers(200, 80, 15);

var data = mlContext.Data.LoadFromEnumerable<Customer>(customers);

var splits = mlContext.Data.TrainTestSplit(data, testFraction:0.2, seed:5);

var pipeline = mlContext.Transforms.Concatenate("Features", nameof(Customer.AnualIncomeK), nameof(Customer.SpendingScore)).
                Append(mlContext.Transforms.NormalizeMinMax("Features")).
                Append(mlContext.Clustering.Trainers.KMeans("Features", numberOfClusters: 3));

var model = pipeline.Fit(splits.TrainSet);

var transform = model.Transform(splits.TestSet);

var evaluation = mlContext.Clustering.Evaluate(transform, scoreColumnName: "Score", featureColumnName: "Features");

Console.WriteLine(evaluation.AverageDistance);
Console.WriteLine(evaluation.DaviesBouldinIndex);

var engine = mlContext.Model.CreatePredictionEngine<Customer, CustomerPrediction>(model);

var predict = engine.Predict(new Customer
{
    AnualIncomeK = 50,
    SpendingScore = 70
});

Console.WriteLine($"Clustered Id is :{predict.ClustedId}");


//low income - high spender
//high income - high spender
//high income - low spender

public class Customer
{
    public float AnualIncomeK { get; set; }
    public float SpendingScore { get; set; }
}

public class CustomerPrediction
{
    [ColumnName("PredictedLabel")]
    public uint ClustedId { get; set; }
    [ColumnName("Score")]
    public float[] Distances { get; set; }
}