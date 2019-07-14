using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace RegressaoComML
{
    class Program
    {
        static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);
            IDataView dataView = mlContext.Data.LoadFromTextFile<TaxiTrip>
                ("taxi-fare-test.csv", hasHeader: true, separatorChar: ',');
            var pipeline = mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "FareAmount")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding
                    (outputColumnName: "VendorIdEncoded", inputColumnName: "VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding
                    (outputColumnName: "RateCodeEncoded", inputColumnName: "RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding
                    (outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mlContext.Transforms.Concatenate
                    ("Features", "VendorIdEncoded", "RateCodeEncoded", "PassengerCount", "TripTime", "TripDistance", "PaymentTypeEncoded"))
                .Append(mlContext.Regression.Trainers.FastTree());
            var model = pipeline.Fit(dataView);
            var predictionFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(model);
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };
            var prediction = predictionFunction.Predict(taxiTripSample);

            Console.WriteLine(prediction.FareAmount);
            Console.ReadKey();
        }
    }
}
