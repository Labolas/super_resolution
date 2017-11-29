#include <iostream>
#include <vector>
#include <visp/vpDebug.h>
#include <visp/vpImage.h>
#include <visp/vpImageIo.h>
#include <visp/vpDisplayX.h>

using namespace std ;

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
inline unsigned char 
getpixelB(const vpImage<vpRGBa>& in, unsigned y, unsigned x)
{
  int h=in.getHeight(), w=in.getWidth();
    if (x < w && y < h)
        return in[y][x].B;

    return 0;
}

static void
bicubicresize(const vpImage<vpRGBa>& in, vpImage<vpRGBa> & out)
{
  int h=in.getHeight(), w=in.getHeight();
  int out_h=out.getHeight(), out_w=out.getWidth();
  
  const float tx = float(w) / out_w;
  const float ty = float(h) / out_h;

  unsigned char C[5] = { 0 };

    for (int i = 0; i < out_h; ++i)
    {
       for (int j = 0; j < out_w; ++j)
       {
         const int x = int(tx * j);
         const int y = int(ty * i);
         const float dx = tx * j - x;
         const float dy = ty * i - y;

         for (int jj = 0; jj < 4; ++jj)
         {
           const int z = y - 1 + jj;
           unsigned char a0 = getpixelR(in, z, x);
           unsigned char d0 = getpixelR(in, z, x - 1) - a0;
           unsigned char d2 = getpixelR(in, z, x + 1) - a0;
           unsigned char d3 = getpixelR(in, z, x + 2) - a0;
           unsigned char a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
           unsigned char a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
           unsigned char a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
           C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

           d0 = C[0] - C[1];
           d2 = C[2] - C[1];
           d3 = C[3] - C[1];
           a0 = C[1];
           a1 = -1.0 / 3 * d0 + d2 -1.0 / 6 * d3;
           a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
           a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
           out[i][j].R = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;
         }
          
         
         for (int jj = 0; jj < 4; ++jj)
         {
           const int z = y - 1 + jj;
           unsigned char a0 = getpixelG(in, z, x);
           unsigned char d0 = getpixelG(in, z, x - 1) - a0;
           unsigned char d2 = getpixelG(in, z, x + 1) - a0;
           unsigned char d3 = getpixelG(in, z, x + 2) - a0;
           unsigned char a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
           unsigned char a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
           unsigned char a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
           C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

           d0 = C[0] - C[1];
           d2 = C[2] - C[1];
           d3 = C[3] - C[1];
           a0 = C[1];
           a1 = -1.0 / 3 * d0 + d2 -1.0 / 6 * d3;
           a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
           a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
           out[i][j].G = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;
         }
          
         for (int jj = 0; jj < 4; ++jj)
         {
           const int z = y - 1 + jj;
           unsigned char a0 = getpixelB(in, z, x);
           unsigned char d0 = getpixelB(in, z, x - 1) - a0;
           unsigned char d2 = getpixelB(in, z, x + 1) - a0;
           unsigned char d3 = getpixelB(in, z, x + 2) - a0;
           unsigned char a1 = -1.0 / 3 * d0 + d2 - 1.0 / 6 * d3;
           unsigned char a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
           unsigned char a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
           C[jj] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;

           d0 = C[0] - C[1];
           d2 = C[2] - C[1];
           d3 = C[3] - C[1];
           a0 = C[1];
           a1 = -1.0 / 3 * d0 + d2 -1.0 / 6 * d3;
           a2 = 1.0 / 2 * d0 + 1.0 / 2 * d2;
           a3 = -1.0 / 6 * d0 - 1.0 / 2 * d2 + 1.0 / 6 * d3;
           out[i][j].B = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;
         }     
       }
    }
}


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
upscale(const vpImage<vpRGBa> &LR, vpImage<vpRGBa> &HR, const unsigned int &N)
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

static void
downscale(const vpImage<vpRGBa> &HR, vpImage<vpRGBa> &LR, const unsigned int &L)
{
  int h=LR.getHeight(), w=LR.getWidth();
  
  // TODO: downscale HR to LR
}



static void
createDico(const vpImage<unsigned char> &comp, vector<unsigned char> * Dl, vector<unsigned char> * Dh)
{
  int h=comp.getHeight(), w=comp.getWidth();

  // dans une dizaine d'images, passage VGG16
  // récupérations de cartes intéressantes (conv2-1, conv2-2)
  
  // ajout de chaque carte sélectionnée dans les dictionnaires Dh et Dl
  
}

static void
Reconstruction(/* arguments */)
{
	/* code */
	//On a Image BF
	//On  Bicubique
	//On vgg16 le resultat de ça
	//On obtient des cartes de features
	//On sélectionne un patch dans l'image et donc aussi dans les cartes de features
	//On sélectionne le meilleur vecteur du dico correspondant à notre vecteur actuel
	//Selon le coef de correlation(=prod scal) plus il est grand mieux c'est

}

int main()
{
  int h=319, w=480;
  int n=2;
  vpImage<vpRGBa> I_LR(h,w,0);
  vpImage<vpRGBa> I_HR(h*n,w*n,0);

  vector<unsigned char> dico[256];
  
  vpImage<unsigned char> Y_HR (h*n,w*n);
  vpImage<unsigned char> Cb_HR(h*n,w*n);
  vpImage<unsigned char> Cr_HR(h*n,w*n);

	vpImage<unsigned char> Y_LR (h,w);
  vpImage<unsigned char> Cb_LR(h,w);
  vpImage<unsigned char> Cr_LR(h,w);

  vpImageIo::read(I_LR,"../img/lion.jpg") ;

  bicubicresize(I_LR, I_HR);
  
  // convertion to YUV
  RGBtoYUV(I_LR, Y_LR, Cb_LR, Cr_LR);

  vpDisplayX d1(I_LR) ;
  vpDisplayX d2(I_HR) ;
  vpDisplayX d3(Y_LR) ;
  vpDisplay::display(I_LR) ;
  vpDisplay::display(I_HR) ;
  vpDisplay::display(Y_LR) ;
  vpDisplay::flush(I_LR) ;
  vpDisplay::flush(I_HR) ;
  vpDisplay::flush(Y_LR) ;
  vpDisplay::getClick(I_LR) ;


  return 0;
}
