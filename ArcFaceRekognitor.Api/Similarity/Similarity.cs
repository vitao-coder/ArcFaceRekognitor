using System.Numerics;

namespace ArcFaceRekognitor.Api.Similarity
{
    public static class Similarity
    {
        public static double CosineSimilarity(float[] vector1, float[] vector2)
        {

            double dotProduct = 0.0;
            double magnitudeA = 0.0;
            double magnitudeB = 0.0;

            for (int i = 0; i < vector1.Length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
                magnitudeA += System.Math.Pow(vector1[i], 2);
                magnitudeB += System.Math.Pow(vector2[i], 2);
            }

            return dotProduct / (System.Math.Sqrt(magnitudeA) * System.Math.Sqrt(magnitudeB));
        }


        //threshold 1.24f
        public static double EuclideanDistance(float[] vector1, float[] vector2)
        {
            double distance = 0.0;
            for (int i = 0; i < vector1.Length; i++)
                distance += System.Math.Pow(vector1[i] - vector2[i], 2);
            return System.Math.Sqrt(distance);
        }
    }
}
