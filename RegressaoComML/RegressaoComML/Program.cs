using System;
using System.Collections.Generic;
using Microsoft.ML;

namespace RegressaoComML
{
    class Program
    {
        static void Main(string[] args)
        {
            // Tem que ter aproximadamente mais de 20 registros de dados
            IEnumerable<Frete> lista = new List<Frete>
            {
                new Frete {Tipo="NORMAL", Distancia = 10, Valor = 20},
                new Frete {Tipo="NORMAL", Distancia = 11, Valor = 21},
                new Frete {Tipo="NORMAL", Distancia = 12, Valor = 22},
                new Frete {Tipo="NORMAL", Distancia = 13, Valor = 23},
                new Frete {Tipo="NORMAL", Distancia = 14, Valor = 24},
                new Frete {Tipo="NORMAL", Distancia = 15, Valor = 25},
                new Frete {Tipo="NORMAL", Distancia = 16, Valor = 26},
                new Frete {Tipo="NORMAL", Distancia = 17, Valor = 27},
                new Frete {Tipo="NORMAL", Distancia = 18, Valor = 28},
                new Frete {Tipo="NORMAL", Distancia = 19, Valor = 29},
                new Frete {Tipo="NORMAL", Distancia = 20, Valor = 30},
                new Frete {Tipo="NORMAL", Distancia = 21, Valor = 31},
                new Frete {Tipo="NORMAL", Distancia = 22, Valor = 32},
                new Frete {Tipo="NORMAL", Distancia = 23, Valor = 33},
                new Frete {Tipo="NORMAL", Distancia = 24, Valor = 34},
                new Frete {Tipo="NORMAL", Distancia = 25, Valor = 35},
                new Frete {Tipo="NORMAL", Distancia = 26, Valor = 36},
                new Frete {Tipo="NORMAL", Distancia = 27, Valor = 37},
                new Frete {Tipo="NORMAL", Distancia = 28, Valor = 38},
                new Frete {Tipo="NORMAL", Distancia = 29, Valor = 39},
                new Frete {Tipo="NORMAL", Distancia = 30, Valor = 40},
                new Frete {Tipo="NORMAL", Distancia = 31, Valor = 41},
                new Frete {Tipo="NORMAL", Distancia = 32, Valor = 42},
                new Frete {Tipo="NORMAL", Distancia = 33, Valor = 43},
                new Frete {Tipo="NORMAL", Distancia = 34, Valor = 44},
                new Frete {Tipo="NORMAL", Distancia = 35, Valor = 45},
                new Frete {Tipo="NORMAL", Distancia = 36, Valor = 46},

                new Frete {Tipo="SEDEX", Distancia = 31, Valor = 51},
                new Frete {Tipo="SEDEX", Distancia = 32, Valor = 52},
                new Frete {Tipo="SEDEX", Distancia = 33, Valor = 53},
                new Frete {Tipo="SEDEX", Distancia = 34, Valor = 54},
                new Frete {Tipo="SEDEX", Distancia = 35, Valor = 55},
                new Frete {Tipo="SEDEX", Distancia = 36, Valor = 56},
                new Frete {Tipo="SEDEX", Distancia = 37, Valor = 57},
                new Frete {Tipo="SEDEX", Distancia = 38, Valor = 58},
                new Frete {Tipo="SEDEX", Distancia = 39, Valor = 59},
                new Frete {Tipo="SEDEX", Distancia = 40, Valor = 60},
                new Frete {Tipo="SEDEX", Distancia = 41, Valor = 61},
                new Frete {Tipo="SEDEX", Distancia = 42, Valor = 62},
            };

            MLContext mlContext = new MLContext(seed: 0);
            IDataView dataView = mlContext.Data.LoadFromEnumerable(lista);

            var pipeline = mlContext.Transforms
                .CopyColumns(outputColumnName: "Label", inputColumnName: "Valor")
                .Append(mlContext.Transforms.Categorical
                    .OneHotEncoding(outputColumnName: "TipoEncoded", inputColumnName: "Tipo"))
                .Append(mlContext.Transforms
                    .Concatenate("Features", "Distancia", "Valor"))
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);

            //var predictions = model.Transform(dataView);
            //var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");
            //Console.WriteLine($"Quanto mais perto de 1, melhor; Score: ${metrics.RSquared:0.##}");

            var predictionFunction = mlContext.Model.CreatePredictionEngine<Frete, PredicaoDaFrete>(model);
            var taxiTripSample = new Frete()
            {
                Tipo = "NORMAL",
                Distancia = 40,
                Valor = 0 // ← valor que será prevido
            };
            var prediction = predictionFunction.Predict(taxiTripSample);
            Console.WriteLine(prediction.Valor);
        }
    }
}
