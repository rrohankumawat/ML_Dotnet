using Microsoft.ML;

var mlContext = new MLContext(seed: 0);

var random = new Random(5);

Purchase NormalPurchase()
{
    float items = random.Next(1, 11);
    return new Purchase
    {
        Items = items,
        Amount = items * 12 + random.Next(-10, 10),
        WeightKg = items * 0.8f + (float)(random.NextDouble() - 0.5)
    };
}

var trainRows = Enumerable.Range(0, 500).Select(_ => NormalPurchase()).ToList();

var testRows = Enumerable.Range(0, 200).Select(_ => NormalPurchase()).ToList();
testRows.Add(new Purchase { Items = 1, Amount = 900, WeightKg = 0.8f });   
testRows.Add(new Purchase { Items = 2, Amount = 700, WeightKg = 1.6f });   
testRows.Add(new Purchase { Items = 10, Amount = 4, WeightKg = 8.0f });   

IDataView trainData = mlContext.Data.LoadFromEnumerable(trainRows);
IDataView testData = mlContext.Data.LoadFromEnumerable(testRows);

var pipeline = mlContext.Transforms.Concatenate("Features", nameof(Purchase.Items), nameof(Purchase.Amount),
    nameof(Purchase.WeightKg)).Append(mlContext.Transforms.NormalizeMinMax("Features")).
    Append(mlContext.AnomalyDetection.Trainers.RandomizedPca("Features", rank: 1));


var model = pipeline.Fit(trainData);

var purchaseScore = mlContext.Data.CreateEnumerable<PurchaseResult>(model.Transform(trainData), reuseRowObject: false).
    Select(x => x.Score).OrderBy(x => x).ToArray();


var cutoff = purchaseScore[(int)(purchaseScore.Length * 0.99)];


var result = mlContext.Data.CreateEnumerable<PurchaseResult>(model.Transform(testData), reuseRowObject: false);

Console.WriteLine($"Cut Off: {cutoff}");
Console.WriteLine($"Average score for test purchase : {result.Average(x=>x.Score)}");


foreach(var purchase in result.OrderByDescending(x => x.Score).Take(5))
{
    Console.WriteLine($"Items: {purchase.Items}, Amount : {purchase.Amount}, " +
        $"WeightKgs {purchase.WeightKg}, Score : {purchase.Score}");
}

public class Purchase
{
    public float Items { get; set; }
    public float Amount { get; set; }
    public float WeightKg { get; set; }
}

public class PurchaseResult
{
    public float Items { get; set; }
    public float Amount { get; set; }
    public float WeightKg { get; set; }
    public float Score { get; set; }
}
