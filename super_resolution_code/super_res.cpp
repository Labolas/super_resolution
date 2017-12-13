#include <iostream>
#include <vector>
#include <string>
#include <visp/vpDebug.h>
#include <visp/vpImage.h>
#include <visp/vpImageIo.h>
#include <visp/vpDisplayX.h>

using namespace std ;

typedef vpRGBa vpYCbCr;

#define BICUBIC     1
#define BILINEAR    0

void Python_Features(vpImage<unsigned char> &I, const char* path);

static void
RGBtoYUV_Double(const vpImage<vpRGBa> &RGB, vpImage<double> &Y_D, vpImage<double> &Cb_D, vpImage<double> &Cr_D)
{
  int h=RGB.getHeight(), w=RGB.getWidth();

  for(int i=0; i<h; i++)
    for(int j=0; j<w; j++)
    {
      Y_D[i][j]  =   0.2125 * RGB[i][j].R + 0.7154 * RGB[i][j].G + 0.0721 * RGB[i][j].B;
      Cb_D[i][j] = - 0.115  * RGB[i][j].R - 0.385  * RGB[i][j].G + 0.5    * RGB[i][j].B + 128;
      Cr_D[i][j] =   0.5    * RGB[i][j].R - 0.454  * RGB[i][j].G - 0.046  * RGB[i][j].B + 128;
    }
}



static void
RGBtoYUV(const vpImage<vpRGBa> &I,
	 vpImage<unsigned char> &Y, vpImage<unsigned char> &Cb, vpImage<unsigned char> &Cr)
{
  int h=I.getHeight(), w=I.getWidth();

  for(int i=0; i<h; i++)
    for(int j=0; j<w; j++)
    {
      Y[i][j]  =   0.2125 * I[i][j].R + 0.7154 * I[i][j].G + 0.0721 * I[i][j].B;
      Cb[i][j] = - 0.115  * I[i][j].R - 0.385  * I[i][j].G + 0.5    * I[i][j].B + 128;
      Cr[i][j] =   0.5    * I[i][j].R - 0.454  * I[i][j].G - 0.046  * I[i][j].B + 128;
    }
}

static void
vpYCbCr_to_double(const vpImage<vpYCbCr> &I,
	 vpImage<double> &Y, vpImage<double> &Cb, vpImage<double> &Cr)
{
  int h=I.getHeight(), w=I.getWidth();

  for(int i=0; i<h; i++)
    for(int j=0; j<w; j++)
    {
      Y[i][j]  =   	(double)I[i][j].R;
      Cb[i][j] = 		(double)I[i][j].G;
      Cr[i][j] =   	(double)I[i][j].B;
    }
}
static void
vpYCbCr_to_RGB(const vpImage<vpYCbCr> &I, vpImage<vpRGBa> &res)
{
  int h=I.getHeight(), w=I.getWidth();

  for(int i=0; i<h; i++)
    for(int j=0; j<w; j++)
    {
      double R  =   	I[i][j].R + (1.4065 * ( I[i][j].B - 128));
      double G  = 	I[i][j].R - (0.3455 * ( I[i][j].G -128)) - (0.7169 * (I[i][j].B - 128));
      double B  =   	I[i][j].R + (1.7790 * ( I[i][j].G - 128));

if(R<0) R = 0; else if (R>255) R = 255;
if(G<0) G = 0; else if (G>255) G = 255;
if(B<0) B = 0; else if (B>255) B = 255;

res[i][j].R = (unsigned char)(floor(R));
res[i][j].G = (unsigned char)(floor(G));
res[i][j].B = (unsigned char)(floor(B));
    }



}

#if BICUBIC
inline unsigned char
getpixelR(const vpImage<vpRGBa>& in, unsigned y, unsigned x)
{
  int h=in.getHeight(), w=in.getWidth();
    if (x < w && y < h)
        return in[y][x].R;

    return 0;
}
inline unsigned char
getpixelG(const vpImage<vpRGBa>& in, unsigned y, unsigned x)
{
  int h=in.getHeight(), w=in.getWidth();
    if (x < w && y < h)
        return in[y][x].G;

    return 0;
}
inline double
getpixelB(const vpImage<vpRGBa>& in, unsigned int y, unsigned int x)
{
  int h=in.getHeight(), w=in.getWidth();
    if (x < w && y < h)
        return (double)(in[y][x].B);

    return 0.0;
}

static void
bicubicresize(const vpImage<vpRGBa>& in, vpImage<vpRGBa> & out)
{
  int h=in.getHeight(), w=in.getWidth();
  int out_h=out.getHeight(), out_w=out.getWidth();

  const double tx = double(w) / out_w;
  const double ty = double(h) / out_h;

  double C[5] = { 0 };

    for (int i = 0; i < out_h; ++i)
    {
       for (int j = 0; j < out_w; ++j)
       {
         const int x = int(tx * j);
         const int y = int(ty * i);
         const double dx = tx * j - x;
         const double dy = ty * i - y;

         for (int jj = 0; jj < 4; ++jj)
         {
           const int z = y - 1 + jj;
           double a0 = getpixelR(in, z, x);
           double d0 = getpixelR(in, z, x - 1) - a0;
           double d2 = getpixelR(in, z, x + 1) - a0;
           double d3 = getpixelR(in, z, x + 2) - a0;
           double a1 = -1.0 / 3.0 * d0 + d2 - 1.0 / 6.0 * d3;
           double a2 = 1.0 / 2.0 * d0 + 1.0 / 2.0 * d2;
           double a3 = -1.0 / 6.0 * d0 - 1.0 / 2.0 * d2 + 1.0 / 6.0 * d3;
           C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

           d0 = C[0] - C[1];
           d2 = C[2] - C[1];
           d3 = C[3] - C[1];
           a0 = C[1];
           a1 = -1.0 / 3.0 * d0 + d2 -1.0 / 6.0 * d3;
           a2 = 1.0 / 2.0 * d0 + 1.0 / 2.0 * d2;
           a3 = -1.0 / 6.0 * d0 - 1.0 / 2.0 * d2 + 1.0 / 6.0 * d3;

           double tmp = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

           if(tmp>255) tmp=255.0;
           if(tmp<0) tmp=0.0;
           out[i][j].R = (unsigned char)(tmp);
         }


         for (int jj = 0; jj < 4; ++jj)
         {
           const int z = y - 1 + jj;
           double a0 = getpixelG(in, z, x);
           double d0 = getpixelG(in, z, x - 1) - a0;
           double d2 = getpixelG(in, z, x + 1) - a0;
           double d3 = getpixelG(in, z, x + 2) - a0;
           double a1 = -1.0 / 3.0 * d0 + d2 - 1.0 / 6.0 * d3;
           double a2 = 1.0 / 2.0 * d0 + 1.0 / 2.0 * d2;
           double a3 = -1.0 / 6.0 * d0 - 1.0 / 2.0 * d2 + 1.0 / 6.0 * d3;
           C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

           d0 = C[0] - C[1];
           d2 = C[2] - C[1];
           d3 = C[3] - C[1];
           a0 = C[1];
           a1 = -1.0 / 3.0 * d0 + d2 -1.0 / 6.0 * d3;
           a2 = 1.0 / 2.0 * d0 + 1.0 / 2.0 * d2;
           a3 = -1.0 / 6.0 * d0 - 1.0 / 2.0 * d2 + 1.0 / 6.0 * d3;

           double tmp = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

           if(tmp>255) tmp=255.0;
           if(tmp<0) tmp=0.0;
           out[i][j].G = (unsigned char)(tmp);
         }

         for (int jj = 0; jj < 4; ++jj)
         {
           const int z = y - 1 + jj;
           double a0 = getpixelB(in, z, x);
           double d0 = getpixelB(in, z, x - 1) - a0;
           double d2 = getpixelB(in, z, x + 1) - a0;
           double d3 = getpixelB(in, z, x + 2) - a0;
           double a1 = -1.0 / 3.0 * d0 + d2 - 1.0 / 6.0 * d3;
           double a2 = 1.0 / 2.0 * d0 + 1.0 / 2.0 * d2;
           double a3 = -1.0 / 6.0 * d0 - 1.0 / 2.0 * d2 + 1.0 / 6.0 * d3;
           C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

           d0 = C[0] - C[1];
           d2 = C[2] - C[1];
           d3 = C[3] - C[1];
           a0 = C[1];
           a1 = -1.0 / 3.0 * d0 + d2 -1.0 / 6.0 * d3;
           a2 =  1.0 / 2.0 * d0 + 1.0 / 2.0 * d2;
           a3 = -1.0 / 6.0 * d0 - 1.0 / 2.0 * d2 + 1.0 / 6.0 * d3;

           double tmp = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;

           if(tmp>255) tmp=255.0;
           if(tmp<0) tmp=0.0;
           out[i][j].B = (unsigned char)(tmp);
         }
       }
    }
}
#endif

#if BILINEAR
/**
 * Interpolation bilineaire
 * @param I : image agrandie incomplete
 * @param p : un point de l image agrandie
 * role: determine la couleur du pixel par interpolation bilineaire
 */
vpRGBa bilinearInterpol(const vpImage<vpRGBa> &I,
			const unsigned int &N,
			const unsigned int &i,
			const unsigned int &j){

  // revoir /////////////////////////////
  int jGauche=0, jDroit=0, iHaut, iBas;
  int h=I.getHeight(), w= I.getWidth();
  if(j%N == 0){
    jGauche=jDroit=j;
  }
  else{
    if(j<=w-N){
      jGauche=round((double)j/N-0.5)*N;
      jDroit=round((double)j/N+0.5)*N;
    }
    else{
      jGauche=jDroit=w-N;
    }
  }
  if(i%N==0){
    iHaut=iBas=i;
  }
  else{
    if(i<=h-N){
      iBas=round((double)i/N-0.5)*N;
      iHaut=round((double)i/N+0.5)*N;
    }
    else{
      iHaut=iBas=h-N;
    }
  }



  unsigned char Red, Green, Blue;
  int tmp=abs(N-(j-jGauche)); // écart pixel gauche
  int tmp2=abs(N-(jDroit-j)); // écart pixel droit
  Red=(unsigned char)((double)(I[iHaut][jGauche].R*tmp+I[iHaut][jDroit].R*tmp2)/(tmp+tmp2));
  Green=(unsigned char)((double)(I[iHaut][jGauche].G*tmp+I[iHaut][jDroit].G*tmp2)/(tmp+tmp2));
  Blue=(unsigned char)((double)(I[iHaut][jGauche].B*tmp+I[iHaut][jDroit].B*tmp2)/(tmp+tmp2));
  vpRGBa x1(Red, Green, Blue);


  Red=(unsigned char)((double)(I[iBas][jGauche].R*tmp+I[iBas][jDroit].R*tmp2)/(tmp+tmp2));
  Green=(unsigned char)((double)(I[iBas][jGauche].G*tmp+I[iBas][jDroit].G*tmp2)/(tmp+tmp2));
  Blue=(unsigned char)((double)(I[iBas][jGauche].B*tmp+I[iBas][jDroit].B*tmp2)/(tmp+tmp2));
  vpRGBa x2(Red, Green, Blue);


  tmp=abs(N-(iHaut-i)); // écart haut
  tmp2=abs(N-(i-iBas)); // écart bas
  Red=(unsigned char)((double)(tmp*x1.R+tmp2*x2.R)/(tmp+tmp2));
  Green=(unsigned char)((double)(tmp*x1.G+tmp2*x2.G)/(tmp+tmp2));
  Blue=(unsigned char)((double)(tmp*x1.B+tmp2*x2.B)/(tmp+tmp2));

  return vpRGBa(Red, Green, Blue);
}



/**
 * Agrandissement par interpolation bilineaire
 * @param compL: Image de base (LR)
 * @param compH: Image de sortie (HR*)
 * @param N: facteur d agrandissement
 */
static void
upscale_bilinearInterpol(const vpImage<vpRGBa> &LR, vpImage<vpRGBa> &HR, const unsigned int &N)
{
  int h=LR.getHeight(), w= LR.getWidth();

  for(int i=0; i<h; i++){
    for(int j=0; j<w; j++){
      HR[i*N][j*N]=LR[i][j];
    }
  }

  for(int i=0; i<h*N; i++){
    for(int j=0; j<w*N; j++){
      HR[i][j]=bilinearInterpol(HR, N, i, j);
    }
  }
}
#endif

static void
completeDico(vector<vpImage<vpYCbCr> > & Dl, vector<vpImage<vpYCbCr> > & Dh)
{
  string img_path= "../data/out/";

  string sY_LR = "lion_Y_LR/conv2/";
  string sCb_LR = "lion_Cb_LR/conv2/";
  string sCr_LR = "lion_Cr_LR/conv2/";

  string sY_HR = "lion_Y_HR/conv2/";
  string sCb_HR = "lion_Cb_HR/conv2/";
  string sCr_HR = "lion_Cr_HR/conv2/";

  int conv2Length = 127;

  for(int a=1; a<3; a++)
  {

    // Y LR
    for(int i=0; i<conv2Length; i++)
    {
      char img_endPath[40];
      sprintf(img_endPath, "%d_conv2_%d.pgm", a, i);

      string path = img_path + sY_LR + img_endPath;

      vpImage<unsigned char> I;
      vpImageIo::read(I,path) ;

      int h=I.getHeight(), w=I.getWidth();

      Dl[i] = vpImage<vpYCbCr>(h,w);

      for(int y=0; y<h; y++)
      {
        for(int x=0; x<w; x++)
        {
          ((Dl[i])[y][x]).R=I[y][x];
        }
      }
    }


    // Cb LR
    for(int i=0; i<conv2Length; i++)
    {
      char img_endPath[40];
      sprintf(img_endPath, "%d_conv2_%d.pgm", a, i);

      string path = img_path + sCb_LR + img_endPath;

      vpImage<unsigned char> I;
      vpImageIo::read(I,path) ;

      int h=I.getHeight(), w=I.getWidth();

      for(int y=0; y<h; y++)
      {
        for(int x=0; x<w; x++)
        {
          ((Dl[i])[y][x]).G=I[y][x];
        }
      }
    }

    // Cr LR
    for(int i=0; i<conv2Length; i++)
    {
      char img_endPath[40];
      sprintf(img_endPath, "%d_conv2_%d.pgm", a, i);

      string path = img_path + sCr_LR + img_endPath;

      vpImage<unsigned char> I;
      vpImageIo::read(I,path) ;

      int h=I.getHeight(), w=I.getWidth();

      for(int y=0; y<h; y++)
      {
        for(int x=0; x<w; x++)
        {
          ((Dl[i])[y][x]).B=I[y][x];
        }
      }
    }


    // Y HR
    for(int i=0; i<conv2Length; i++)
    {
      char img_endPath[40];
      sprintf(img_endPath, "%d_conv2_%d.pgm", a, i);

      string path = img_path + sY_HR + img_endPath;

      vpImage<unsigned char> I;
      vpImageIo::read(I,path) ;

      int h=I.getHeight(), w=I.getWidth();


      Dl[i] = vpImage<vpYCbCr>(h,w);

      for(int y=0; y<h; y++)
      {
        for(int x=0; x<w; x++)
        {
          ((Dl[i])[y][x]).R=I[y][x];
        }
      }
    }


    // Cb HR
    for(int i=0; i<conv2Length; i++)
    {
      char img_endPath[40];
      sprintf(img_endPath, "%d_conv2_%d.pgm", a, i);

      string path = img_path + sCb_HR + img_endPath;

      vpImage<unsigned char> I;
      vpImageIo::read(I,path) ;

      int h=I.getHeight(), w=I.getWidth();

      for(int y=0; y<h; y++)
      {
        for(int x=0; x<w; x++)
        {
          ((Dl[i])[y][x]).G=I[y][x];
        }
      }
    }



    // Cr HR
    for(int i=0; i<conv2Length; i++)
    {
      char img_endPath[40];
      sprintf(img_endPath, "%d_conv2_%d.pgm", a, i);

      string path = img_path + sCr_HR + img_endPath;

      vpImage<unsigned char> I;
      vpImageIo::read(I,path) ;

      int h=I.getHeight(), w=I.getWidth();

      for(int y=0; y<h; y++)
      {
        for(int x=0; x<w; x++)
        {
          ((Dl[i])[y][x]).B=I[y][x];
        }
      }
    }

  }

}

static void
createDico(vector<vpImage<vpYCbCr> > & Dl, vector<vpImage<vpYCbCr> > & Dh)
{
  vpImage<vpYCbCr> cartesLR, cartesHR;

  // dans une dizaine d'images, passage VGG16

  // récupérations de cartes intéressantes (conv2-1, conv2-2)

  // ajout de chaque carte sélectionnée dans les dictionnaires Dh et Dl







  // pour l'instant, récupération de toutes les cartes:


  // resize factor
  int n=2;

  // Low resolution image
  vpImage<vpRGBa> I_HR;
  vpImageIo::read(I_HR,"../data/img/lion.jpg") ;
  int h=I_HR.getHeight(), w=I_HR.getWidth();

  vpImage<unsigned char> Y_HR(h,w);
  vpImage<unsigned char> Cb_HR(h,w);
  vpImage<unsigned char> Cr_HR(h,w);

  vpImage<unsigned char> Y_LR(h,w);
  vpImage<unsigned char> Cb_LR(h,w);
  vpImage<unsigned char> Cr_LR(h,w);

  RGBtoYUV(I_HR, Y_HR, Cb_HR, Cr_HR);

  // VGG16 on HR image
  //Python_Features(Y_HR, "lion_Y_HR");
  //Python_Features(Cb_HR, "lion_Cb_HR");
  //Python_Features(Cr_HR, "lion_Cr_HR");

  // Low Resolution Image
  vpImage<vpRGBa> I_LR(h/n,w/n,0);
  vpImage<vpRGBa> I_HRbis(h,w,0);

  // Resize
  bicubicresize(I_HR, I_LR);
  bicubicresize(I_LR, I_HRbis);

  RGBtoYUV(I_HRbis, Y_LR, Cb_LR, Cr_LR);

  // VGG16 on LR image

  //Python_Features(Y_LR, "lion_Y_LR");
  //Python_Features(Cb_LR, "lion_Cb_LR");
  //Python_Features(Cr_LR, "lion_Cr_LR");

  // copy maps into dictionaries
  completeDico(Dl, Dh);

}
/////////////////////////////////////////////////
//////////////Reconstrution Thibault
/////////////////////////////////////////////////
void
Python_Features(vpImage<unsigned char> &I, const char* path) {
	string imgPath = "../data/img/";
	vpImageIo::write(I,imgPath+path+".jpg");
  char python[30];
  sprintf(python,"python CAV.py %s.jpg",path)  ;
	system(python); 	//On vgg16 le resultat de ça
}

static void CalculMoyennePatch(vpImage<vpYCbCr> &I, vpImage<unsigned char> &res,
    vpImage<double> & ecartType) {

  int h_HR = I.getHeight();
  int w_HR = I.getWidth();

  int compteur = 0; //compteur pour la moyenne
  double sumY = 0;
  double variance = 0;
	for(int i = 0 ; i<h_HR; i++)
	{
		for (int j = 0; j<w_HR; j++)
		{

      sumY = 0;
      compteur = 0;
      variance = 0;
			for(int ii = -4 ; ii<5; ii++)
			{
				for (int jj = -4; jj<5; jj++)
				{
					if(ii+i >= 0 && ii+i < h_HR && jj+j >= 0 && jj+j < w_HR)
					{
						sumY	 += I[ii+i][jj+j].R;
						compteur++;
					}
				}
			}

      
			double moyPatchY 	= sumY  / compteur;
			res[i][j] =  moyPatchY;

      for(int ii = -4 ; ii<5; ii++)
			{
				for (int jj = -4; jj<5; jj++)
				{
					if(ii+i >= 0 && ii+i < h_HR && jj+j >= 0 && jj+j < w_HR)
					{
            variance += (I[ii+i][jj+j].R - moyPatchY) * (I[ii+i][jj+j].R - moyPatchY) ;
					}
				}
			}
      
      
      
      variance /= compteur;
      
      ecartType[i][j] = sqrt(variance);
      
      
		}
	}
}

static void
PatchManager(vpImage<vpRGBa> &HR, vpImage<double> & ecartType1,
	vpImage<unsigned char> &resY, vpImage<unsigned char> &resCb,vpImage<unsigned char> &resCr) {

	int h_HR = HR.getHeight();
	int w_HR = HR.getWidth();

	vpImage<unsigned char> hrY(h_HR,w_HR);
  vpImage<unsigned char> hrCb(h_HR,w_HR);
  vpImage<unsigned char> hrCr(h_HR,w_HR);
	RGBtoYUV(HR,hrY,hrCb,hrCr);

	//On sélectionne un patch dans l'image et donc aussi dans les cartes de features
	int compteur = 0; //compteur pour la moyenne
	double sumY = 0;double sumCb = 0;double sumCr = 0;
  double variance = 0;

	for(int i = 0 ; i<h_HR; i++)
	{
		for (int j = 0; j<w_HR; j++)
		{
      sumY =0; sumCb = 0; sumCr = 0;
      compteur = 0;
      variance = 0;
			for(int ii = -4 ; ii<5; ii++)
			{
				for (int jj = -4; jj<5; jj++)
				{
					if(ii+i >= 0 && ii+i < h_HR && jj+j >= 0 && jj+j < w_HR)
					{
						sumY	 += hrY[ii+i][jj+j];
						sumCb	 += hrCb[ii+i][jj+j];
						sumCr	 += hrCr[ii+i][jj+j];
						compteur ++;
					}
				}
			}

			double moyPatchY 	= sumY  / compteur;
			double moyPatchCb = sumCb / compteur;
			double moyPatchCr = sumCr / compteur;

      resY[i][j] 	=  moyPatchY;
      resCb[i][j] =  moyPatchCb;
      resCr[i][j] =  moyPatchCr;

      for(int ii = -4 ; ii<5; ii++)
			{
				for (int jj = -4; jj<5; jj++)
				{
					if(ii+i >= 0 && ii+i < h_HR && jj+j >= 0 && jj+j < w_HR)
					{
            variance += (hrY[ii+i][jj+j] - moyPatchY) * (hrY[ii+i][jj+j]- moyPatchY) ;
					}
				}
			}
      variance /= compteur;
      ecartType1[i][j] = sqrt(variance);
		}
	}
}

static void
DicoVectorSelection(vector<vpImage<vpYCbCr> > dicoLR, vector<vpImage<vpYCbCr> > dicoHR,
	vpImage<unsigned char> &resY, vpImage<unsigned char> &resCb, vpImage<unsigned char> &resCr,
  vpImage<double> ecartType1, vpImage<vpRGBa> &HR, vpImage<vpRGBa> &resultat) {

    int h_HR = HR.getHeight();
    int w_HR = HR.getWidth();
    vpImage<unsigned char> hrY(h_HR,w_HR);
    vpImage<unsigned char> hrCb(h_HR,w_HR);
    vpImage<unsigned char> hrCr(h_HR,w_HR);
  	RGBtoYUV(HR,hrY,hrCb,hrCr);
vpImage<vpYCbCr> resYCbCr (h_HR,w_HR);

  int h = dicoLR[0].getHeight(), w = dicoLR[0].getWidth();
  vpImage<vpYCbCr> elementDico(h,w);

  vpImage<unsigned char> Imoy(h,w,0);
  vpImage<double> indexY(h,w,0);
  double meilleurValY = 0;
  double produitScalY = 0;
  vpImage<double> ecartType2(h_HR,w_HR);

  for (int s = 0; s<256 ; s++)
  {
    CalculMoyennePatch(dicoLR[s], Imoy, ecartType2);
    for(int i = 0 ; i<h; i++)
    {
      for (int j = 0; j<w; j++)
      {
        produitScalY = 0;
        for(int ii = -4 ; ii<5; ii++)
        {
          for (int jj = -4; jj<5; jj++)
          {
            if(ii+i >= 0 && ii+i < h && jj+j >= 0 && jj+j < w)
            {
              produitScalY  += (hrY[ii+i][jj+j] - resY[i][j]) * (dicoLR[s][ii+i][jj+j].R -Imoy[i][j]);
              //produitScalCb += dicoLR[s][ii+i][jj+j].G * resCb[ii+i][jj+j];
              //produitScalCr += dicoLR[s][ii+i][jj+j].B * resCr[ii+i][jj+j];
            }
          }
        }

        if(ecartType1[i][j] == 0 ) ecartType1[i][j] =1;
        if(ecartType2[i][j] == 0 ) ecartType2[i][j] =1;
        
        produitScalY /= ecartType1[i][j]*ecartType2[i][j];
        
        if(produitScalY > meilleurValY)
        {
          
          cout << "test1" <<endl;
          meilleurValY = produitScalY;
          indexY[i][j] = s;
          cout << "test2" <<endl;
        }
        
      }
    }
  }
  
  
  for(int i = 0 ; i<h; i++)
    {
      for (int j = 0; j<w; j++)
      {
	resYCbCr[i][j].R = dicoHR[indexY[i][j]][i][j].R;
	resYCbCr[i][j].G = dicoHR[indexY[i][j]][i][j].G;
	resYCbCr[i][j].B = dicoHR[indexY[i][j]][i][j].B;
      }
   }
   vpYCbCr_to_RGB(resYCbCr,resultat);
}


static void
Reconstruction(vpImage<vpRGBa> &LR, vpImage<vpRGBa> &HR,
  vector<vpImage<vpYCbCr> > dicoLR,vector<vpImage<vpYCbCr> > dicoHR)
{

	int h = HR.getHeight();
	int w = HR.getWidth();

	vpImage<vpRGBa> resultat(h,w);

	vpImage<unsigned char> featureY(h,w);
	vpImage<unsigned char> featureCb(h,w);
	vpImage<unsigned char> featureCr(h,w);
  	vpImage<double> ecartType1(h,w);

	bicubicresize(LR, HR); // HR est l'image agrandi BF (bicubique ou lineaire interpol)

	//Python_Features(featureY,"Reconst_HR_Y"); //On obtient des cartes de features
  //Python_Features(featureCb,"Reconst_HR_Cb"); //On obtient des cartes de features
  //Python_Features(featureCr,"Reconst_HR_Cr"); //On obtient des cartes de features

  //system("python CAV.py lion.jpg"); 	//On vgg16 le resultat de ça

	PatchManager(HR, ecartType1, featureY,featureCb,featureCr);

	//On sélectionne le meilleur vecteur du dico correspondant à notre vecteur actuel
	DicoVectorSelection(dicoLR,dicoHR, featureY, featureCb,featureCr, ecartType1, HR,resultat);

	//garder le coef de correlation

	//save
	vpImageIo::write(resultat,"../data/img/superRes.jpg") ;

}

int main()
{
  // resize factor
  int n=2;

  vector<vpImage<vpYCbCr> > dicoLR(256);
  vector<vpImage<vpYCbCr> > dicoHR(256);
  
  cout << "Dictionary: Init" << endl;
  
  createDico(dicoLR,dicoHR);
  
  cout << "Dictionary : Done" << endl;
  

  vpImage<vpRGBa> I_LR;
  vpImageIo::read(I_LR,"../data/img/lionReconst_LR.jpg") ;
  int h=I_LR.getHeight(), w=I_LR.getWidth();
  vpImage<vpRGBa> I_HR(h*2,w*2);
  

  cout << "Reconstruction: Init" << endl;
  
  Reconstruction(I_LR,I_HR,dicoLR,dicoHR);
  
  cout << "Reconstruction: Init" << endl;
  return 0;
}
