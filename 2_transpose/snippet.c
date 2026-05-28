for (i = 0; i < N; i+=8)
    {
        for (j = 0; j < N; j+=8)
        {
            for(k=i;k<i+4;k++){
                t0=A[k][j];t1=A[k][j+1];t2=A[k][j+2];t3=A[k][j+3];
                t4=A[k][j+4];t5=A[k][j+5];t6=A[k][j+6];t7=A[k][j+7];

                B[j][k]=t0;
                B[j+1][k]=t1;
                l=B[j][k];
                B[j+2][k]=t2;
                l=B[j+1][k];
                B[j+3][k]=t3;

                B[j][k+4]=t4;
                B[j+1][k+4]=t5;
                l=B[j][k+4];
                B[j+2][k+4]=t6;
                l=B[j+1][k+4];
                B[j+3][k+4]=t7;
            }
            for(k=j;k<j+4;k++){
                t0=B[k][i+4];t1=B[k][i+5];t2=B[k][i+6];t3=B[k][i+7];
                t4=A[i+4][k];t5=A[i+5][k];t6=A[i+6][k];t7=A[i+7][k];

                B[k][i+4]=t4;
                B[k][i+5]=t5;
                B[k][i+6]=t6;
                B[k][i+7]=t7;

                if(k==j) l=B[j+2][i+4];
                if(k==j+1) l=B[j+3][i+4];
                if(k==j+2) l=B[j+4][i];
                if(k==j+3) l=B[j+5][i];

                B[k+4][i]=t0;
                B[k+4][i+1]=t1;
                B[k+4][i+2]=t2;
                B[k+4][i+3]=t3;

            }
            for(k=i+4;k<i+8;k++){
                t0=A[k][j+4];t1=A[k][j+5];t2=A[k][j+6];t3=A[k][j+7];

                B[j+4][k]=t0;
                B[j+5][k]=t1;
                l=B[j+4][k];
                B[j+6][k]=t2;
                l=B[j+5][k];
                B[j+7][k]=t3;
            }
        }
    }