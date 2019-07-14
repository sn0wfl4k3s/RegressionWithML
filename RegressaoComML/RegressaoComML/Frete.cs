using Microsoft.ML.Data;

namespace RegressaoComML
{
    public class Frete
    {
        [LoadColumn(0)]
        public string Tipo { get; set; }
        [LoadColumn(1)]
        public float Distancia { get; set; }
        [LoadColumn(2)]
        public float Valor { get; set; }
    }

    public class PredicaoDaFrete
    {
        [ColumnName("Score")]
        public float Valor { get; set; }
    }
}
