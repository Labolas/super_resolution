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

void Python_Features(vpImage<unsigned char> & I, const char* path);

static void
RGBtoYUV_Double(const vpImage<vpRGBa> & RGB, vpImage<double> & Y_D, vpImage<double> & Cb_D, vpImage<double> & Cr_D)
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
RGBtoYUV(const vpImage<vpRGBa> & I,
	 vpImage<unsigned char> & Y, vpImage<unsigned char> & Cb, vpImage<unsigned char> & Cr)
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
vpYCbCr_to_double(const vpImage<vpYCbCr> & I,
	 vpImage<double> & Y, vpImage<double> & Cb, vpImage<double> & Cr)
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
vpYCbCr_to_RGB(const vpImage<vpYCbCr> & I, vpImage<vpRGBa> & res)
{
  int h=I.getHeight(), w=I.getWidth();

  for(int i=0; i<h; i++)
    for(int j=0; j<w; j++)
    {
      double R  =   	I[i][j].R + (1.4065 * ( I[i][j].B - 128));
      double G  = 	  I[i][j].R - (0.3455 * ( I[i][j].G - 128)) - (0.7169 * (I[i][j].B - 128));
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
getpixelR(const vpImage<vpRGBa> & in, const unsigned & y, const unsigned & x)
{
  int h=in.getHeight(), w=in.getWidth();
    if (x < w && y < h)
        return in[y][x].R;

    return 0;
}
inline unsigned char
getpixelG(const vpImage<vpRGBa> & in, const unsigned & y, const unsigned & x)
{
  int h=in.getHeight(), w=in.getWidth();
    if (x < w && y < h)
        return in[y][x].G;

    return 0;
}
inline double
getpixelB(const vpImage<vpRGBa> & in, const unsigned int & y, const unsigned int & x)
{
  int h=in.getHeight(), w=in.getWidth();
    if (x < w && y < h)
        return (double)(in[y][x].B);

    return 0.0;
}

static void
bicubicresize(const vpImage<vpRGBa> & in, vpImage<vpRGBa> & out)
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
vpRGBa bilinearInterpol(const vpImage<vpRGBa> & I,
			const unsigned int & N,
			const unsigned int & i,
			const unsigned int & j)
{
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
upscale_bilinearInterpol(const vpImage<vpRGBa> & LR, vpImage<vpRGBa> & HR, const unsigned int & N)
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

  int conv2Length = 128;

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

      Dl[i+128*(a-1)] = vpImage<vpYCbCr>(h,w);

      for(int y=0; y<h; y++)
      {
        for(int x=0; x<w; x++)
        {
          ((Dl[i+128*(a-1)])[y][x]).R=I[y][x];
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
          ((Dl[i+128*(a-1)])[y][x]).G=I[y][x];
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
          ((Dl[i+128*(a-1)])[y][x]).B=I[y][x];
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


      Dh[i+128*(a-1)] = vpImage<vpYCbCr>(h,w);

      for(int y=0; y<h; y++)
      {
        for(int x=0; x<w; x++)
        {
          ((Dh[i+128*(a-1)])[y][x]).R=I[y][x];
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
          ((Dh[i+128*(a-1)])[y][x]).G=I[y][x];
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
          ((Dh[i+128*(a-1)])[y][x]).B=I[y][x];
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
  vpImage<vpRGBa> I_base_LR;
  vpImage<vpRGBa> I_base_HR;
  vpImageIo::read(I_base_LR,"../data/img/lionReconst_LR.png") ;
  vpImageIo::read(I_base_HR,"../data/img/lion.png") ;
  int h=I_base_HR.getHeight(), w=I_base_HR.getWidth();
  
  vpImage<vpRGBa> I_LR_bicu(h, w, 0);
  
  bicubicresize(I_base_LR, I_LR_bicu);

  vpImage<unsigned char> Y_HR(h,w);
  vpImage<unsigned char> Cb_HR(h,w);
  vpImage<unsigned char> Cr_HR(h,w);

  vpImage<unsigned char> Y_LR(h,w);
  vpImage<unsigned char> Cb_LR(h,w);
  vpImage<unsigned char> Cr_LR(h,w);

  RGBtoYUV(I_LR_bicu, Y_LR, Cb_LR, Cr_LR);
  RGBtoYUV(I_base_HR, Y_HR, Cb_HR, Cr_HR);

  // VGG16 on HR image
  //Python_Features(Y_LR, "lion_Y_LR");
  //Python_Features(Cb_LR, "lion_Cb_LR");
  //Python_Features(Cr_LR, "lion_Cr_LR");
  
  // VGG16 on LR image
  //Python_Features(Y_HR, "lion_Y_HR");
  //Python_Features(Cb_HR, "lion_Cb_HR");
  //Python_Features(Cr_HR, "lion_Cr_HR");

  // copy maps into dictionaries
  completeDico(Dl, Dh);

}
/////////////////////////////////////////////////
//////////////Reconstrution Thibault
/////////////////////////////////////////////////
void
Python_Features(vpImage<unsigned char> & I, const char* path) {
	string imgPath = "../data/img/";
	vpImageIo::write(I,imgPath+path+".png");
  char python[30];
  sprintf(python,"python CAV.py %s.png",path)  ;
	system(python); 	//On vgg16 le resultat de ça
}

static void CalculMoyennePatch(vpImage<vpYCbCr> & I, vpImage<double> & ImoyY, vpImage<double> & ImoyCb, vpImage<double> & ImoyCr,
    vpImage<double> & ecartTypeY, vpImage<double> & ecartTypeCb,vpImage<double> & ecartTypeCr, const int & s)
{

  int h = I.getHeight();
  int w = I.getWidth();

  int compteur = 0; //compteur pour la moyenne
  double sumY = 0;
  double sumCb = 0;
  double sumCr = 0;
  double varianceY = 0;
  double varianceCb = 0;
  double varianceCr = 0;
	for(int i = 0 ; i<h; i++)
	{
		for (int j = 0; j<w; j++)
		{
      sumY  = 0;
      sumCb = 0;
      sumCr = 0;
      compteur = 0;
      varianceY  = 0;
      varianceCb = 0;
      varianceCr = 0;
			for(int ii = -2 ; ii<3; ii++)
			{
				for (int jj = -2; jj<3; jj++)
				{
					if(ii+i >= 0 && ii+i < h && jj+j >= 0 && jj+j < w)
					{
						sumY	 += I[ii+i][jj+j].R;
						sumCb	 += I[ii+i][jj+j].G;
						sumCr	 += I[ii+i][jj+j].B;
						compteur++;
					}
				}
			}

			double moyPatchY  	= sumY   / compteur;
			double moyPatchCb 	= sumCb  / compteur;
			double moyPatchCr 	= sumCr  / compteur;
			
      ImoyY[i][j]  =  moyPatchY;
			ImoyCb[i][j] =  moyPatchCb;
			ImoyCr[i][j] =  moyPatchCr;

      for(int ii = -2 ; ii<3; ii++)
			{
				for (int jj = -2; jj<3; jj++)
				{
					if(ii+i >= 0 && ii+i < h && jj+j >= 0 && jj+j < w)
					{
            varianceY  += (I[ii+i][jj+j].R - moyPatchY)  * (I[ii+i][jj+j].R - moyPatchY) ;
            varianceCb += (I[ii+i][jj+j].G - moyPatchCb) * (I[ii+i][jj+j].G - moyPatchCb) ;
            varianceCr += (I[ii+i][jj+j].B - moyPatchCr) * (I[ii+i][jj+j].B - moyPatchCr) ;
					}
				}
			}
           
      varianceY  /= compteur;
      varianceCb /= compteur;
      varianceCr /= compteur;
      
      ecartTypeY[i][j]  = sqrt(varianceY);
      ecartTypeCb[i][j] = sqrt(varianceCb);
      ecartTypeCr[i][j] = sqrt(varianceCr);
		}
	}
}

static void
PatchManager(vpImage<vpRGBa> & HR, vpImage<double> & ecartType1Y, vpImage<double> & ecartType1Cb, vpImage<double> & ecartType1Cr,
	vpImage<double> & moy1Y, vpImage<double> & moy1Cb, vpImage<double> & moy1Cr) {

	int h_HR = HR.getHeight();
	int w_HR = HR.getWidth();

	vpImage<unsigned char> hrY(h_HR,w_HR);
  vpImage<unsigned char> hrCb(h_HR,w_HR);
  vpImage<unsigned char> hrCr(h_HR,w_HR);
	RGBtoYUV(HR,hrY,hrCb,hrCr);

	//On sélectionne un patch dans l'image et donc aussi dans les cartes de features
	double compteur = 0; //compteur pour la moyenne
	double sumY = 0;double sumCb = 0;double sumCr = 0;
  double varianceY = 0;
  double varianceCb = 0;
  double varianceCr = 0;

	for(int i = 0 ; i<h_HR; i++)
	{
		for (int j = 0; j<w_HR; j++)
		{
      sumY =0; sumCb = 0; sumCr = 0;
      compteur = 0;
      varianceY = 0;
      varianceCb = 0;
      varianceCr = 0;
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

      moy1Y[i][j]  =  moyPatchY;
      moy1Cb[i][j] =  moyPatchCb;
      moy1Cr[i][j] =  moyPatchCr;

      for(int ii = -4 ; ii<5; ii++)
			{
				for (int jj = -4; jj<5; jj++)
				{
					if(ii+i >= 0 && ii+i < h_HR && jj+j >= 0 && jj+j < w_HR)
					{
            varianceY  += (hrY[ii+i][jj+j]  - moyPatchY)  * (hrY[ii+i][jj+j]  - moyPatchY) ;
            varianceCb += (hrCb[ii+i][jj+j] - moyPatchCb) * (hrCb[ii+i][jj+j] - moyPatchCb) ;
            varianceCr += (hrCr[ii+i][jj+j] - moyPatchCr) * (hrCr[ii+i][jj+j] - moyPatchCr) ;
					}
				}
			}
      varianceY  /= compteur;
      varianceCb /= compteur;
      varianceCr /= compteur;
      
      ecartType1Y[i][j]  = sqrt(varianceY);
      ecartType1Cb[i][j] = sqrt(varianceCb);
      ecartType1Cr[i][j] = sqrt(varianceCr);
		}
	}
}

static void
DicoVectorSelection(vector<vpImage<vpYCbCr> > & dicoLR, vector<vpImage<vpYCbCr> > & dicoHR,
	vpImage<double> & moy1Y, vpImage<double> & moy1Cb, vpImage<double> & moy1Cr,
  vpImage<double> & ecartType1Y, vpImage<double> & ecartType1Cb, vpImage<double> & ecartType1Cr, 
  vpImage<vpRGBa> & HR, vpImage<vpRGBa> & resultat) {

    int h_HR = HR.getHeight();
    int w_HR = HR.getWidth();
    vpImage<unsigned char> hrY(h_HR,w_HR);
    vpImage<unsigned char> hrCb(h_HR,w_HR);
    vpImage<unsigned char> hrCr(h_HR,w_HR);
  	RGBtoYUV(HR,hrY,hrCb,hrCr);
    vpImage<vpYCbCr> resYCbCr (h_HR,w_HR);

  int h = dicoLR[0].getHeight(), w = dicoLR[0].getWidth();
  vpImage<vpYCbCr> elementDico(h,w);
  
  vector<vpImage<double> > ImoyY(256);
  vector<vpImage<double> > ImoyCb(256);
  vector<vpImage<double> > ImoyCr(256);
  
  vpImage<int> indexY(h_HR,w_HR,0);
  vpImage<int> indexCb(h_HR,w_HR,0);
  vpImage<int> indexCr(h_HR,w_HR,0);
  
  vpImage<int> index2Y(h_HR,w_HR,0);
  vpImage<int> index2Cb(h_HR,w_HR,0);
  vpImage<int> index2Cr(h_HR,w_HR,0);
  
  vpImage<int> index3Y(h_HR,w_HR,0);
  vpImage<int> index3Cb(h_HR,w_HR,0);
  vpImage<int> index3Cr(h_HR,w_HR,0);
  
  vpImage<double> coefY(h_HR,w_HR,0);
  vpImage<double> coefCb(h_HR,w_HR,0);
  vpImage<double> coefCr(h_HR,w_HR,0);
  
  vpImage<double> coef2Y(h_HR,w_HR,0);
  vpImage<double> coef2Cb(h_HR,w_HR,0);
  vpImage<double> coef2Cr(h_HR,w_HR,0);
  
  vpImage<double> coef3Y(h_HR,w_HR,0);
  vpImage<double> coef3Cb(h_HR,w_HR,0);
  vpImage<double> coef3Cr(h_HR,w_HR,0);
  
  double meilleurValY = 0;
  double produitScalY = 0;
  double meilleurValCb = 0;
  double produitScalCb = 0;
  double meilleurValCr = 0;
  double produitScalCr = 0;
  vector<vpImage<double> > ecartType2Y(256);
  vector<vpImage<double> > ecartType2Cb(256);
  vector<vpImage<double> > ecartType2Cr(256);

  for(int s=0; s<256; s++)
  {
    ImoyY[s] = vpImage<double>(h,w);
    ImoyCb[s] = vpImage<double>(h,w);
    ImoyCr[s] = vpImage<double>(h,w);
    ecartType2Y[s]  = vpImage<double>(h,w,0);
    ecartType2Cb[s] = vpImage<double>(h,w,0);
    ecartType2Cr[s] = vpImage<double>(h,w,0);
    CalculMoyennePatch(dicoLR[s], ImoyY[s], ImoyCb[s], ImoyCr[s], ecartType2Y[s], ecartType2Cb[s], ecartType2Cr[s], s);
  }
  
  cout << "means & standard deviation : done" << endl;
  
    for(int i = 0 ; i<h_HR-2; i++)
    {
      for (int j = 0; j<w_HR; j++)
      {
        meilleurValY = 0;
        meilleurValCb = 0;
        meilleurValCr = 0;
        for (int s = 0; s<256 ; s++)
        {
          produitScalY = 0;
          produitScalCb = 0;
          produitScalCr = 0;
          int cpt = 0;
          for(int ii = -4 ; ii<5; ii++)
          {
            for (int jj = -4; jj<5; jj++)
            {
              if(ii+i >= 0 && ii+i < h_HR-6 && jj+j >= 0 && jj+j < w_HR)
              {
                produitScalY   += (hrY[ii+i][jj+j]  - moy1Y[i][j])  * (dicoLR[s][(ii+i)>>1][(jj+j)>>1].R - ImoyY[s][i>>1][j>>1]);
                produitScalCb  += (hrCb[ii+i][jj+j] - moy1Cb[i][j]) * (dicoLR[s][(ii+i)>>1][(jj+j)>>1].G - ImoyCb[s][i>>1][j>>1]);
                produitScalCr  += (hrCr[ii+i][jj+j] - moy1Cr[i][j]) * (dicoLR[s][(ii+i)>>1][(jj+j)>>1].B - ImoyCr[s][i>>1][j>>1]);
                //produitScalY  +=(double)(hrY[ii+i][jj+j])/255.0  * (double)(dicoLR[s][(ii+i)>>1][(jj+j)>>1].R)/255.0 ;
                //produitScalCb +=(double)(hrCb[ii+i][jj+j])/255.0 * (double)(dicoLR[s][(ii+i)>>1][(jj+j)>>1].G)/255.0 ;
                //produitScalCr +=(double)(hrCr[ii+i][jj+j])/255.0 * (double)(dicoLR[s][(ii+i)>>1][(jj+j)>>1].B)/255.0 ;
                //double y =(double)(hrY[ii+i][jj+j])/255.0  - (double)(dicoLR[s][(ii+i)>>1][(jj+j)>>1].R)/255.0 ;
                //double cb=(double)(hrCb[ii+i][jj+j])/255.0 - (double)(dicoLR[s][(ii+i)>>1][(jj+j)>>1].G)/255.0 ;
                //double cr=(double)(hrCr[ii+i][jj+j])/255.0 - (double)(dicoLR[s][(ii+i)>>1][(jj+j)>>1].B)/255.0 ;
                //produitScalY   += y*y;
                //produitScalCb  += cb*cb;
                //produitScalCr  += cr*cr;
                //produitScalY  += abs((double)(dicoLR[s][(ii+i)>>1][(jj+j)>>1].R)/255.0 - (double)(hrY[ii+i][jj+j])/255.0) ;
                //produitScalCb += abs((double)(dicoLR[s][(ii+i)>>1][(jj+j)>>1].G)/255.0 - (double)(hrCb[ii+i][jj+j])/255.0) ;
                //produitScalCr += abs((double)(dicoLR[s][(ii+i)>>1][(jj+j)>>1].B)/255.0 - (double)(hrCr[ii+i][jj+j])/255.0) ;
                cpt++;
              }
            }
          }

          //produitScalY  = sqrt(produitScalY);
          //produitScalCb = sqrt(produitScalCb);
          //produitScalCr = sqrt(produitScalCr);
          
          produitScalY  /= cpt;
          produitScalCb /= cpt;
          produitScalCr /= cpt;
          
          if(ecartType1Y[i][j]  == 0 ) ecartType1Y[i][j]  = 1;
          if(ecartType1Cb[i][j] == 0 ) ecartType1Cb[i][j] = 1;
          if(ecartType1Cr[i][j] == 0 ) ecartType1Cr[i][j] = 1;
          
          if(ecartType2Y[s][i>>1][j>>1]  == 0 ) ecartType2Y[s][i>>1][j>>1]  = 1;
          if(ecartType2Cb[s][i>>1][j>>1] == 0 ) ecartType2Cb[s][i>>1][j>>1] = 1;
          if(ecartType2Cr[s][i>>1][j>>1] == 0 ) ecartType2Cr[s][i>>1][j>>1] = 1;
          
        
          produitScalY  /= ecartType1Y[i][j]  * ecartType2Y[s][i>>1][j>>1];
          produitScalCb /= ecartType1Cb[i][j] * ecartType2Cb[s][i>>1][j>>1];
          produitScalCr /= ecartType1Cr[i][j] * ecartType2Cr[s][i>>1][j>>1];
     
          if(produitScalY > meilleurValY)
          {
            meilleurValY = produitScalY;
            indexY[i][j] = s;
            coefY[i][j] = meilleurValY;
          }  
          if(produitScalCb > meilleurValCb)
          {
            meilleurValCb = produitScalCb;
            indexCb[i][j] = s;
            coefCb[i][j] = meilleurValCb;
          }  
          if(produitScalCr > meilleurValCr)
          {
            meilleurValCr = produitScalCr;
            indexCr[i][j] = s;
            coefCr[i][j] = meilleurValCr;
          }  
        }
        
        // --------------------------------------
        meilleurValY = 0;
        meilleurValCb = 0;
        meilleurValCr = 0;
        for (int s = 0; s<256 ; s++)
        {
          produitScalY = 0;
          produitScalCb = 0;
          produitScalCr = 0;
          int cpt = 0;
          for(int ii = -4 ; ii<5; ii++)
          {
            for (int jj = -4; jj<5; jj++)
            {
              if(ii+i >= 0 && ii+i < h_HR-6 && jj+j >= 0 && jj+j < w_HR)
              {
                produitScalY   += (hrY[ii+i][jj+j]  - moy1Y[i][j])  * (dicoLR[s][(ii+i)>>1][(jj+j)>>1].R - ImoyY[s][i>>1][j>>1]);
                produitScalCb  += (hrCb[ii+i][jj+j] - moy1Cb[i][j]) * (dicoLR[s][(ii+i)>>1][(jj+j)>>1].G - ImoyCb[s][i>>1][j>>1]);
                produitScalCr  += (hrCr[ii+i][jj+j] - moy1Cr[i][j]) * (dicoLR[s][(ii+i)>>1][(jj+j)>>1].B - ImoyCr[s][i>>1][j>>1]);
                cpt++;
              }
            }
          }

          produitScalY  /= cpt;
          produitScalCb /= cpt;
          produitScalCr /= cpt;
          
          if(ecartType1Y[i][j]  == 0 ) ecartType1Y[i][j]  = 1;
          if(ecartType1Cb[i][j] == 0 ) ecartType1Cb[i][j] = 1;
          if(ecartType1Cr[i][j] == 0 ) ecartType1Cr[i][j] = 1;
          
          if(ecartType2Y[s][i>>1][j>>1]  == 0 ) ecartType2Y[s][i>>1][j>>1]  = 1;
          if(ecartType2Cb[s][i>>1][j>>1] == 0 ) ecartType2Cb[s][i>>1][j>>1] = 1;
          if(ecartType2Cr[s][i>>1][j>>1] == 0 ) ecartType2Cr[s][i>>1][j>>1] = 1;
          
        
          produitScalY  /= ecartType1Y[i][j]  * ecartType2Y[s][i>>1][j>>1];
          produitScalCb /= ecartType1Cb[i][j] * ecartType2Cb[s][i>>1][j>>1];
          produitScalCr /= ecartType1Cr[i][j] * ecartType2Cr[s][i>>1][j>>1];
         
          if(s != indexY[i][j] && produitScalY > meilleurValY)
          {
            meilleurValY = produitScalY;
            index2Y[i][j] = s;
            coef2Y[i][j] = meilleurValY;
          }  
          if(s != indexCb[i][j] && produitScalCb > meilleurValCb)
          {
            meilleurValCb = produitScalCb;
            index2Cb[i][j] = s;
            coef2Cb[i][j] = meilleurValCb;
          }  
          if(s != indexCr[i][j] && produitScalCr > meilleurValCr)
          {
            meilleurValCr = produitScalCr;
            index2Cr[i][j] = s;
            coef2Cr[i][j] = meilleurValCr;
          }  
        }
        
        // --------------------------------------
        meilleurValY = 0;
        meilleurValCb = 0;
        meilleurValCr = 0;
        for (int s = 0; s<256 ; s++)
        {
          produitScalY = 0;
          produitScalCb = 0;
          produitScalCr = 0;
          int cpt = 0;
          for(int ii = -4 ; ii<5; ii++)
          {
            for (int jj = -4; jj<5; jj++)
            {
              if(ii+i >= 0 && ii+i < h_HR-6 && jj+j >= 0 && jj+j < w_HR)
              {
                produitScalY   += (hrY[ii+i][jj+j]  - moy1Y[i][j])  * (dicoLR[s][(ii+i)>>1][(jj+j)>>1].R - ImoyY[s][i>>1][j>>1]);
                produitScalCb  += (hrCb[ii+i][jj+j] - moy1Cb[i][j]) * (dicoLR[s][(ii+i)>>1][(jj+j)>>1].G - ImoyCb[s][i>>1][j>>1]);
                produitScalCr  += (hrCr[ii+i][jj+j] - moy1Cr[i][j]) * (dicoLR[s][(ii+i)>>1][(jj+j)>>1].B - ImoyCr[s][i>>1][j>>1]);
                cpt++;
              }
            }
          }

          produitScalY  /= cpt;
          produitScalCb /= cpt;
          produitScalCr /= cpt;
          
          if(ecartType1Y[i][j]  == 0 ) ecartType1Y[i][j]  = 1;
          if(ecartType1Cb[i][j] == 0 ) ecartType1Cb[i][j] = 1;
          if(ecartType1Cr[i][j] == 0 ) ecartType1Cr[i][j] = 1;
          
          if(ecartType2Y[s][i>>1][j>>1]  == 0 ) ecartType2Y[s][i>>1][j>>1]  = 1;
          if(ecartType2Cb[s][i>>1][j>>1] == 0 ) ecartType2Cb[s][i>>1][j>>1] = 1;
          if(ecartType2Cr[s][i>>1][j>>1] == 0 ) ecartType2Cr[s][i>>1][j>>1] = 1;
          
        
          produitScalY  /= ecartType1Y[i][j]  * ecartType2Y[s][i>>1][j>>1];
          produitScalCb /= ecartType1Cb[i][j] * ecartType2Cb[s][i>>1][j>>1];
          produitScalCr /= ecartType1Cr[i][j] * ecartType2Cr[s][i>>1][j>>1];
     
          if(s != indexY[i][j] && s != index2Y[i][j] && produitScalY > meilleurValY)
          {
            meilleurValY = produitScalY;
            index3Y[i][j] = s;
            coef3Y[i][j] = meilleurValY;
          }  
          if(s != indexCb[i][j] && s != index2Cb[i][j] && produitScalCb > meilleurValCb)
          {
            meilleurValCb = produitScalCb;
            index3Cb[i][j] = s;
            coef3Cb[i][j] = meilleurValCb;
          }  
          if(s != indexCr[i][j] && s != index2Cr[i][j] && produitScalCr > meilleurValCr)
          {
            meilleurValCr = produitScalCr;
            index3Cr[i][j] = s;
            coef3Cr[i][j] = meilleurValCr;
          }  
        }
      }
    }
  
  
  // prendre plusieurs vecteurs et faire une moyenne pondérée 
  
  cout << "indexation done" <<endl;
  vpImage<unsigned char> testY(h_HR, w_HR, 0);
  vpImage<unsigned char> testCb(h_HR, w_HR, 0);
  vpImage<unsigned char> testCr(h_HR, w_HR, 0);
  
  for(int i = 0 ; i<h_HR-6; i++)
    {
      for (int j = 0; j<w_HR; j++)
      { 
        int y  = (dicoHR[indexY[i][j]][i>>1][j>>1].R  * coefY[i][j]  + dicoHR[index2Y[i][j]][i>>1][j>>1].R  * coef2Y[i][j]  + dicoHR[index3Y[i][j]][i>>1][j>>1].R  * coef3Y[i][j]  )/(coefY[i][j]+coef2Y[i][j]+coef3Y[i][j]);
        int cb = (dicoHR[indexCb[i][j]][i>>1][j>>1].G * coefCb[i][j] + dicoHR[index2Cb[i][j]][i>>1][j>>1].G * coef2Cb[i][j] + dicoHR[index3Cb[i][j]][i>>1][j>>1].G * coef3Cb[i][j] )/(coefCb[i][j]+coef2Cb[i][j]+coef3Cb[i][j]);
        int cr = (dicoHR[indexCr[i][j]][i>>1][j>>1].B * coefCr[i][j] + dicoHR[index2Cr[i][j]][i>>1][j>>1].B * coef2Cr[i][j] + dicoHR[index3Cr[i][j]][i>>1][j>>1].B * coef3Cr[i][j] )/(coefCr[i][j]+coef2Cr[i][j]+coef3Cr[i][j]);
          
        if(y<0) y=0;
        if(y>255) y=255;
        if(cb<0) cb=0;
        if(cb>255) cb=255;
        if(cr<0) cr=0;
        if(cr>255) cr=255;
        
        testY[i][j]  = y;
        testCb[i][j] = cb;
        testCr[i][j] = cr; 
	      
	      resYCbCr[i][j].R = y; 
	      resYCbCr[i][j].G = cb;
	      resYCbCr[i][j].B = cr;
  
        //cout << "index = " << (int)indexY[i][j] << " ; " << (int)indexCb[i][j] << " ; " << (int)indexCr[i][j] << "\t\t\tYCbCr = " << (int)resYCbCr[i][j].R << " ; " << (int)resYCbCr[i][j].G << " ; " << (int)resYCbCr[i][j].B << endl; 
      }
   }
  
  
	vpImageIo::write(testY,"../data/img/testY.png") ;
	vpImageIo::write(testCb,"../data/img/testCb.png") ;
	vpImageIo::write(testCr,"../data/img/testCr.png") ;

  
   vpYCbCr_to_RGB(resYCbCr, resultat);
}


static void
Reconstruction(vpImage<vpRGBa> & LR, vpImage<vpRGBa> & HR,
  vector<vpImage<vpYCbCr> > & dicoLR, vector<vpImage<vpYCbCr> > & dicoHR)
{

	int h = HR.getHeight();
	int w = HR.getWidth();

	vpImage<vpRGBa> resultat(h,w);

	vpImage<double> moy1Y(h,w);
	vpImage<double> moy1Cb(h,w);
	vpImage<double> moy1Cr(h,w);
  
	vpImage<unsigned char> HR_Y(h,w);
	vpImage<unsigned char> HR_Cb(h,w);
	vpImage<unsigned char> HR_Cr(h,w);
  
  vpImage<double> ecartType1Y(h,w);
  vpImage<double> ecartType1Cb(h,w);
  vpImage<double> ecartType1Cr(h,w);

	bicubicresize(LR, HR); // HR est l'image agrandi BF (bicubique ou lineaire interpol)
  
  RGBtoYUV(HR, HR_Y, HR_Cb, HR_Cr);

	//Python_Features(HR_Y,"Reconst_HR_Y"); //On obtient des cartes de features
  //Python_Features(HR_Cb,"Reconst_HR_Cb"); //On obtient des cartes de features
  //Python_Features(HR_Cr,"Reconst_HR_Cr"); //On obtient des cartes de features

  //system("python CAV.py lion.png"); 	//On vgg16 le resultat de ça
  
	PatchManager(HR, ecartType1Y, ecartType1Cb, ecartType1Cr, moy1Y, moy1Cb, moy1Cr);
  
	//On sélectionne le meilleur vecteur du dico correspondant à notre vecteur actuel
	DicoVectorSelection(dicoLR, dicoHR, moy1Y, moy1Cb, moy1Cr, ecartType1Y, ecartType1Cb, ecartType1Cr, HR, resultat);

	//garder le coef de correlation

	//save
	vpImageIo::write(resultat,"../data/img/superRes.png") ;

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
  vpImageIo::read(I_LR,"../data/img/lionReconst_LR.png") ;
  int h=I_LR.getHeight(), w=I_LR.getWidth();
  vpImage<vpRGBa> I_HR(h*2,w*2,0);
  

  cout << "Reconstruction: Init" << endl;
  
  Reconstruction(I_LR, I_HR, dicoLR, dicoHR);
  
  cout << "Reconstruction: Done" << endl;
  
  return 0;
}
