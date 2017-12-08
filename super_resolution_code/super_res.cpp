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
completeDico(vector<vpImage<vpYCbCr> > * Dl, vector<vpImage<vpYCbCr> > * Dh, const int & h, const int & w)
{
   string img_path= "../data/out/";
}

static void
createDico(vector<vpImage<vpYCbCr> > * Dl, vector<vpImage<vpYCbCr> > * Dh)
{
  vpImage<vpYCbCr> cartesLR, cartesHR;
  
  // dans une dizaine d'images, passage VGG16

  // récupérations de cartes intéressantes (conv2-1, conv2-2)
  
  // ajout de chaque carte sélectionnée dans les dictionnaires Dh et Dl
  
  
  
  
  
  
  
  // pour l'instant, récupération de toutes les cartes:
  
  
  // resize factor
  int n=2;
  
  // Low resolution image
  vpImage<vpRGBa> I_LR;
  vpImageIo::read(I_LR,"../data/img/lion.jpg") ;
  int h=I_LR.getHeight(), w=I_LR.getWidth();
  
  // High Resolution Image
  vpImage<vpRGBa> I_HR(h*n,w*n,0);
  
  // Resize
  bicubicresize(I_LR, I_HR);
  
  // VGG16 on I_HR
  
  // copy maps into dictionaries
  completeDico(Dl, Dh, h, w);
  
}
/////////////////////////////////////////////////
//////////////Reconstrution Thibault
/////////////////////////////////////////////////
static void
Python_Features(vpImage<vpRGBa> &HR) {
	string imgPath = "../data/img/";
	vpImageIo::write(HR,imgPath+"Reconst_HR.jpg");
	system("python CAV.py Reconst_HR.jpg"); 	//On vgg16 le resultat de ça
}

static void
PatchManager(vpImage<vpRGBa> &HR,
	vpImage<double> &resY, vpImage<double> &resCb,vpImage<double> &resCr) {

	int h_HR = HR.getHeight();
	int w_HR = HR.getWidth();

	vpImage<double> hrY(h_HR,w_HR); vpImage<double> hrCb(h_HR,w_HR); vpImage<double> hrCr(h_HR,w_HR);
	RGBtoYUV_Double(HR,hrY,hrCb,hrCr);

	//On sélectionne un patch dans l'image et donc aussi dans les cartes de features
	int compteur = 0; //compteur pour la moyenne
	double sumY = 0;double sumCb = 0;double sumCr = 0;
	for(int i = 0 ; i<h_HR; i++)
	{
		for (int j = 0; j<w_HR; j++)
		{
			for(int ii = -4 ; ii<5; ii++)
			{
				for (int jj = -4; jj<5; jj++)
				{
					if(ii+i >= 0 || ii+i < h_HR || jj+j >= 0 || jj+j < w_HR)
					{
						sumY	 += hrY[i+ii][j+jj];
						sumCb	 += hrCb[i+ii][j+jj];
						sumCr	 += hrCr[i+ii][j+jj];
						compteur ++;
					}
				}
			}

			double moyPatchY 	= sumY  / compteur;
			double moyPatchCb = sumCb / compteur;
			double moyPatchCr = sumCr / compteur;

			for(int iii = -4 ; iii<5 ; iii++)
			{
				for (int jjj = -4 ; jjj<5; jjj++)
				{
					if(iii+i >= 0 || iii+i < h_HR || jjj+j >= 0 || jjj+j < w_HR)
					{
						resY[i+iii][j+jjj] 	= hrY[i+iii][j+jjj]  - moyPatchY;
						resCb[i+iii][j+jjj] = hrCb[i+iii][j+jjj] - moyPatchCb;
						resCr[i+iii][j+jjj] = hrCr[i+iii][j+jjj] - moyPatchCr;
					}
				}
			}
		}
	}
}

static void
DicoVectorSelection(/*Dico de Basse Res,*/
	vpImage<double> &resY, vpImage<double> &resCb, vpImage<double> &resCr) {
	//caster l'élément du dio en double

}


static void
Reconstruction(vpImage<vpRGBa> &LR, vpImage<vpRGBa> &HR)
{
	int h = HR.getHeight();
	int w = HR.getWidth();

	vpImage<double> resY(h,w);
	vpImage<double> resCb(h,w);
	vpImage<double> resCr(h,w);

	bicubicresize(LR, HR); // HR est l'image agrandi BF (bicubique ou lineaire interpol)

	Python_Features(HR); //On obtient des cartes de features

	PatchManager(HR,resY,resCb,resCr);

	//On sélectionne le meilleur vecteur du dico correspondant à notre vecteur actuel
	DicoVectorSelection(/*dico de LR,*/ resY, resCb,resCr);

	//garder le coef de correlation

}

int main()
{

  // resize factor
  int n=2;
  
  // Low resolution image
  vpImage<vpRGBa> I_LR;
  vpImageIo::read(I_LR,"../data/img/lion.jpg") ;
  int h=I_LR.getHeight(), w=I_LR.getWidth();
  
  // High Resolution Image
  vpImage<vpRGBa> I_HR(h*n,w*n,0);
  
  // Resize
  bicubicresize(I_LR, I_HR);
  
  vpDisplayX d1(I_LR,100,100) ;
  vpDisplayX d2(I_HR,100,100) ;
  vpDisplay::setTitle(I_LR, "original image");
  vpDisplay::setTitle(I_HR, "original image");
  vpDisplay::display(I_LR);
  vpDisplay::display(I_HR);
  vpDisplay::flush(I_LR) ;
  vpDisplay::flush(I_HR) ;	 
  vpDisplay::getClick(I_HR) ;
  
  
  return 0;
}
