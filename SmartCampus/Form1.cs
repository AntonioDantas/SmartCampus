using Accord.Neuro;
using Accord.Neuro.Learning;
using GMap.NET;
using GMap.NET.MapProviders;
using GMap.NET.WindowsForms;
using GMap.NET.WindowsForms.Markers;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.Neural;
using SharpLearning.Neural.Activations;
using SharpLearning.Neural.Layers;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural.Loss;
using SharpLearning.Neural.Models;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;
using SmartCampus.Properties;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SmartCampus
{
    public partial class Form1 : Form
    {
        public List<Dado> dados = new List<Dado>(); //Lista de Informações
        public string[] predios = { "IEST", "IMC", "IEPG", "ADM", "BIM" };
        public int[] quantidades = { 0, 0, 0, 0, 0 }; //Quantidade a ser previsata
        List<List<PointLatLng>> points = new List<List<PointLatLng>>(); //Lista dos pontos para gerar o poligono
        List<PointLatLng> centros = new List<PointLatLng>(); //Lista dos pontos centrais do poligono
        public string treinamento = "D;A0;A1;A2;A3;A4;T\r\n"; //Cabeçalho
        public string teste = "D;A0;A1;A2;A3;A4;T\r\n"; //Cabeçalho
        double[] UltimaObservacao = null; //Auxiliar para previsão até data corrente
        DateTime UltimaHora = DateTime.Now; //Auxiliar para previsão até data corrente
        RegressionNeuralNetModel modelnet; //modelo para treinamento da Rede Neural
        RegressionForestModel model; //modelo para treinamento da Floresta Aleatória
        double trainErrorNet, testErrorNet, trainError, testError = 0; //Variáveis para teste dos modelos

        public Form1()
        {
            InitializeComponent();
            StreamReader rd = new StreamReader(@"..\..\..\example.csv");

            string linha = null;
            string[] linhaseparada = null;
            linha = rd.ReadLine();
            while ((linha = rd.ReadLine()) != null)
            {
                try
                {
                    //linhaseparada = linha.Split(';');
                    linhaseparada = linha.Split(',');
                    string[] dt = linhaseparada[3].Split(' ');
                    string[] sd = linhaseparada[6].Replace("hrs", " ").Replace("min", " ").Replace("sec", "").Replace("  ", " ").Split(' ');

                    dados.Add(new Dado
                    {
                        //Tue May 21 19:12:14 BRT 2019
                        DataHora = Convert.ToDateTime($"{dt[2]}/{dt[1]}/{dt[5]} {dt[3]}"),
                        //1hrs15min 13sec
                        Duracao = new TimeSpan(Convert.ToInt32((sd.Count() == 2) ? "0" : sd[0]), Convert.ToInt32((sd.Count() == 2) ? sd[0] : sd[1]), Convert.ToInt32((sd.Count() == 2) ? sd[1] : sd[2])),
                        MacUsuario = linhaseparada[2],
                        Login = linhaseparada[0],
                        Local = linhaseparada[4]
                    });
                }
                catch { }
            }
            rd.Close();

            //Propriedades do Mapa
            map.MapProvider = GMapProviders.GoogleMap;
            map.Position = new GMap.NET.PointLatLng(-22.4124616, -45.4492968);
            map.MaxZoom = 20;
            map.MinZoom = 5;
            map.Zoom = 17;


            initColorsBlocks();
            Visualizacoes();

            //Todos Prédios
            for (var p = 0; p < predios.Count(); p++)
            {
                try
                {
                    ObterDados(predios[p]);
                    RegressionLearner_Learn_And_Predict();

                    double predicaoAnt = UltimaObservacao[5];
                    double predicao = UltimaObservacao[5];

                    //DateTime objetivo = DateTime.Now.AddHours(1);
                    DateTime objetivo = UltimaHora.AddHours(1);

                    while (UltimaHora < objetivo) //Enquanto não chegar a data atual + 1 hora permanece prevendo
                    {
                        //Estrutura para predição
                        double[] pre = new double[] { UltimaHora.DayOfWeek.GetHashCode(), UltimaObservacao[2], UltimaObservacao[3], UltimaObservacao[4], UltimaObservacao[5], predicaoAnt };

                        if (testErrorNet < testError) //Qual melhor técnica
                            predicao = modelnet.Predict(pre);
                        else
                            predicao = model.Predict(pre);

                        UltimaObservacao = new double[] { UltimaHora.DayOfWeek.GetHashCode(), UltimaObservacao[2], UltimaObservacao[3], UltimaObservacao[4], UltimaObservacao[5], predicaoAnt, predicao };

                        predicaoAnt = predicao;
                        UltimaHora = UltimaHora.AddHours(1);
                    }

                    //Resultado da predição
                    quantidades[p] = (int)predicao;
                }
                catch
                {

                }
            }

            for (int i = 0; i < predios.Count(); i++)
                CriaRetangulo(points[i], predios[i], centros[i], quantidades[i], quantidades.Max());
        }

        public byte Alpha = 0x7d;
        public List<Color> ColorsOfMap = new List<Color>();

        private void initColorsBlocks()
        {
            ColorsOfMap.AddRange(new Color[]{
            Color.FromArgb(Alpha, 0, 0, 0xFF) ,//Blue
            Color.FromArgb(Alpha, 0, 0xFF, 0xFF) ,//Cyan
            Color.FromArgb(Alpha, 0, 0xFF, 0) ,//Green
            Color.FromArgb(Alpha, 0xFF, 0xFF, 0) ,//Yellow
            Color.FromArgb(Alpha, 0xFF, 0, 0) ,//Red
        });
        }

        /// <summary>
        /// Predição de Floresta Aleatória e Rede Neural
        /// </summary>
        public void RegressionLearner_Learn_And_Predict()
        {
            #region Treinamento da Floresta Aleatória
            var parser = new CsvParser(() => new StringReader(treinamento));
            var targetName = "T";

            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();
            UltimaObservacao = new double[] { observations[observations.RowCount - 1, 0], observations[observations.RowCount - 1, 2], observations[observations.RowCount - 1, 3], observations[observations.RowCount - 1, 4], observations[observations.RowCount - 1, 5], targets[targets.Count() - 1] };

            var learner = new RegressionRandomForestLearner(trees: 5000);
            model = learner.Learn(observations, targets);
            #endregion 

            #region Teste da Floresta Aleatória
            parser = new CsvParser(() => new StringReader(teste));
            var observationsTeste = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();
            var targetsTeste = parser.EnumerateRows(targetName)
                           .ToF64Vector();

            // predict the training and test set.
            var trainPredictions = model.Predict(observations);
            var testPredictions = model.Predict(observationsTeste);

            // create the metric
            var metric = new MeanSquaredErrorRegressionMetric();

            // measure the error on training and test set.
            trainError = metric.Error(targets, trainPredictions);
            testError = metric.Error(targetsTeste, testPredictions);
            #endregion

            #region Treinamento da Rede Neural
            var net = new NeuralNet();
            net.Add(new InputLayer(6));
            net.Add(new DropoutLayer(0.2));
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new DenseLayer(800, Activation.Relu));
            net.Add(new DropoutLayer(0.5));
            net.Add(new SquaredErrorRegressionLayer());

            var learnernet = new RegressionNeuralNetLearner(net, iterations: 2000, loss: new SquareLoss());
            modelnet = learnernet.Learn(observations, targets);
            #endregion

            #region Teste da Rede Neural
            trainPredictions = modelnet.Predict(observations);
            testPredictions = modelnet.Predict(observationsTeste);

            trainErrorNet = metric.Error(targets, trainPredictions);
            testErrorNet = metric.Error(targetsTeste, testPredictions);
            #endregion

        }

        public Brush GetColorForValue(double val, double maxVal)
        {
            double valPerc = val / maxVal;// value%
            double colorPerc = 1d / (ColorsOfMap.Count - 1);// % of each block of color. the last is the "100% Color"
            double blockOfColor = valPerc / colorPerc;// the integer part repersents how many block to skip
            int blockIdx = (int)Math.Truncate(blockOfColor);// Idx of 
            double valPercResidual = valPerc - (blockIdx * colorPerc);//remove the part represented of block 
            double percOfColor = valPercResidual / colorPerc;// % of color of this block that will be filled

            Color cTarget = ColorsOfMap[blockIdx];
            Color cNext = new Color();
            try
            {
                cNext = cNext = ColorsOfMap[blockIdx + 1];
            }
            catch
            {
                cNext = cNext = ColorsOfMap[blockIdx];
            }
            var deltaR = cNext.R - cTarget.R;
            var deltaG = cNext.G - cTarget.G;
            var deltaB = cNext.B - cTarget.B;

            var R = cTarget.R + (deltaR * percOfColor);
            var G = cTarget.G + (deltaG * percOfColor);
            var B = cTarget.B + (deltaB * percOfColor);

            Color c = ColorsOfMap[0];
            try
            {
                c = Color.FromArgb(Alpha, (byte)R, (byte)G, (byte)B);
            }
            catch (Exception ex)
            {
            }
            return new SolidBrush(c);
        }

        /// <summary>
        /// Plotar Poligono no Mapa
        /// </summary>
        /// <param name="points">Pontos do poligono</param>
        /// <param name="nome">Nome do Prédio</param>
        /// <param name="label">Ponto Central do Prédio</param>
        /// <param name="qtd">Quantidade de pessoas no Prédio</param>
        /// <param name="max">Quantidade Máxima dos Prédios</param>
        private void CriaRetangulo(List<PointLatLng> points, string nome, PointLatLng label, int qtd, int max)
        {
            if (qtd > 0)
            {
                GMapOverlay polyOverlay = new GMapOverlay("polygons");
                GMapPolygon polygon = new GMapPolygon(points, "mypolygon");
                polygon.Fill = GetColorForValue(qtd, max);
                polygon.Stroke = new Pen(Color.Black, 1);
                polyOverlay.Polygons.Add(polygon);
                polyOverlay.Markers.Add(new GMarkerCross(label) { ToolTipText = nome + ": " + qtd, IsVisible = true, ToolTipMode = MarkerTooltipMode.Always });
                map.Overlays.Add(polyOverlay);
            }
        }

        /// <summary>
        /// Visualização dos resultados no mapa
        /// </summary>
        private void Visualizacoes()
        {
            map.Overlays.Clear();

            List<PointLatLng> lst = new List<PointLatLng>();
            lst.Add(new PointLatLng(-22.412097, -45.450659));
            lst.Add(new PointLatLng(-22.411383, -45.450104));
            lst.Add(new PointLatLng(-22.411936, -45.449243));
            lst.Add(new PointLatLng(-22.412667, -45.449836));
            points.Add(lst);
            centros.Add(new PointLatLng(-22.412081, -45.450002));

            List<PointLatLng> lst2 = new List<PointLatLng>();
            lst2.Add(new PointLatLng(-22.414410, -45.449176));
            lst2.Add(new PointLatLng(-22.413884, -45.448707));
            lst2.Add(new PointLatLng(-22.414244, -45.448074));
            lst2.Add(new PointLatLng(-22.414827, -45.448535));
            points.Add(lst2);
            centros.Add(new PointLatLng(-22.414358, -45.448651));

            List<PointLatLng> lst3 = new List<PointLatLng>();
            lst3.Add(new PointLatLng(-22.413944, -45.450194));
            lst3.Add(new PointLatLng(-22.413203, -45.449615));
            lst3.Add(new PointLatLng(-22.413587, -45.449051));
            lst3.Add(new PointLatLng(-22.414291, -45.449641));
            points.Add(lst3);
            centros.Add(new PointLatLng(-22.413830, -45.449593));

            List<PointLatLng> lst4 = new List<PointLatLng>();
            lst4.Add(new PointLatLng(-22.413215, -45.450666));
            lst4.Add(new PointLatLng(-22.412920, -45.450427));
            lst4.Add(new PointLatLng(-22.413173, -45.449968));
            lst4.Add(new PointLatLng(-22.413438, -45.450186));
            points.Add(lst4);
            centros.Add(new PointLatLng(-22.413210, -45.450381));

            List<PointLatLng> lst5 = new List<PointLatLng>();
            lst5.Add(new PointLatLng(-22.412962, -45.449166));
            lst5.Add(new PointLatLng(-22.412560, -45.448871));
            lst5.Add(new PointLatLng(-22.412893, -45.448318));
            lst5.Add(new PointLatLng(-22.413344, -45.448683));
            points.Add(lst5);
            centros.Add(new PointLatLng(-22.412982, -45.448726));
        }

        private void PictureBox1_Click(object sender, EventArgs e)
        {

        }

        /// <summary>
        /// Método para Obter os dados de determinado prédio
        /// </summary>
        /// <param name="predio">Nome do Prédio para Identificação</param>
        public void ObterDados(string predio)
        {
            treinamento = "D;A0;A1;A2;A3;A4;T\r\n";
            teste = "D;A0;A1;A2;A3;A4;T\r\n";

            List<Dado> l = (from d in dados where d.Local.Contains(predio) select d).OrderBy(x => x.DataHora).ToList();
            try
            {
                //Inicia no mês 5 até mês 6
                for (var m = 5; m <= 6; m++)
                {
                    List<Dado> mes = (from d in l where d.DataHora.Month == m select d).OrderBy(x => x.DataHora).ToList();
                    //Inicia no dia 1 até o dia 31
                    for (var i = 1; i <= 31 && mes.Count > 0; i++)
                    {
                        List<Dado> temp = (from d in mes where d.DataHora.Day == i select d).OrderBy(x => x.DataHora).ToList();
                        //Inicia na hora 5 (janela) até as 23 horas
                        for (var j = 5; j < 24 && temp.Count > 0; j++)
                        {
                            try
                            {
                                //Monta linha do CSV para predição
                                string linha = $"{new DateTime(2019, m, i).DayOfWeek.GetHashCode()};";
                                for (var k = j - 5; k <= j; k++)
                                {
                                    linha += $"{(from x in temp where x.DataHora.Hour >= k && k <= x.DataHora.Add(x.Duracao).Hour select x.MacUsuario).Distinct().Count()};";
                                }
                                linha = linha.Substring(0, linha.Length - 1) + "\r\n";

                                //Se alguma previsão retornou 0
                                if (!linha.Contains(";0"))
                                {
                                    //Separa um range para Treinamento e outro para Teste
                                    if (new DateTime(2019, m, i) < new DateTime(2019, 6, 20))
                                    {
                                        treinamento += linha;
                                        //Atualiza última hora com dados reais
                                        UltimaHora = new DateTime(2019, m, i);
                                    }
                                    else
                                    {
                                        teste += linha;
                                    }
                                }
                            }
                            catch
                            {

                            }
                        }
                    }
                }
            }
            catch (Exception err)
            {

            }

            //Retira espaço vazio
            treinamento = treinamento.Substring(0, treinamento.Length - 2);
            teste = teste.Substring(0, teste.Length - 2);

        }
    }
}
