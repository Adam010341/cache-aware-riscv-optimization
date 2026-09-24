#include<riscv_vector.h>
#define MIN(a,b) ((a<b) ? a:b)
/* ============================================================
 * Computes C = A * B
 *   A : M x K  (row-major)
 *   B : K x N  (row-major)
 *   C : M x N  (row-major)
 * ============================================================ */
void matmul(float *A, float *B, float *C, int M, int K, int N) {
    for(int x=0;x<M*N;x++){
        C[x]=0.0f;
    }

    for(int m0=0;m0<M;m0+=16){
        int n=N;
        int j=0;
        while(n>0){
            size_t vl=__riscv_vsetvl_e32m4(n);
            for(int k0=0;k0<K;k0+=16){
                int i=m0;
                for(;i<=MIN(m0+16,M)-4;i+=4){
                    vfloat32m4_t vc0=__riscv_vle32_v_f32m4(&C[(i+0)*N+j],vl);
                    vfloat32m4_t vc1=__riscv_vle32_v_f32m4(&C[(i+1)*N+j],vl);
                    vfloat32m4_t vc2=__riscv_vle32_v_f32m4(&C[(i+2)*N+j],vl);
                    vfloat32m4_t vc3=__riscv_vle32_v_f32m4(&C[(i+3)*N+j],vl);
                    for(int k=k0;k<MIN(k0+16,K);k++){
                        vfloat32m4_t vb=__riscv_vle32_v_f32m4(&B[k*N+j],vl);

                        vc0=__riscv_vfmacc_vf_f32m4(vc0, A[(i+0)*K+k],vb,vl);
                        vc1=__riscv_vfmacc_vf_f32m4(vc1, A[(i+1)*K+k],vb,vl);
                        vc2=__riscv_vfmacc_vf_f32m4(vc2, A[(i+2)*K+k],vb,vl);
                        vc3=__riscv_vfmacc_vf_f32m4(vc3, A[(i+3)*K+k],vb,vl);
                    }
                    __riscv_vse32_v_f32m4(&C[(i+0)*N+j],vc0,vl);
                    __riscv_vse32_v_f32m4(&C[(i+1)*N+j],vc1,vl);
                    __riscv_vse32_v_f32m4(&C[(i+2)*N+j],vc2,vl);
                    __riscv_vse32_v_f32m4(&C[(i+3)*N+j],vc3,vl);
                }
                for(;i<MIN(m0+16,M);i++){
                    vfloat32m4_t vc0=__riscv_vle32_v_f32m4(&C[(i+0)*N+j],vl);
                    for(int k=k0;k<MIN(k0+16,K);k++){
                        vfloat32m4_t vb=__riscv_vle32_v_f32m4(&B[k*N+j],vl);

                        vc0=__riscv_vfmacc_vf_f32m4(vc0, A[(i+0)*K+k],vb,vl);
                    }
                    __riscv_vse32_v_f32m4(&C[(i+0)*N+j],vc0,vl);
                }
            }
            n-=vl;
            j+=vl;
        }
    }
}