//#define INDEX(X) clamp(X.x, 0, sizes[0] - 1)*sizes[1] + clamp(X.y, 0, sizes[1] - 1)
#define INDEX(X) (X.x % sizes[0])*sizes[1] + (X.y % sizes[1])
#define dt 0.01f

#define C_coof 1.f
#define ENa 50.f
#define EK -77.f
#define EL -54.6f
#define gNa 120.f
#define gK 36.f
#define gL 0.3f
#define conductivity 20.0f

#define max_image_V 70.0f
#define min_image_V -70.0f

float alpha_n(float V){
    return 0.01*(V-10.0f)/(1.0f-exp((10.0f-V)/10.0f));
}
float beta_n(float V){
    return 0.125*exp(-V/80.0f);
}
float alpha_m(float V){
    return 0.1f*(V-25.f)/(1-exp((25.f-V)/10.f));
}
float beta_m(float V){
    return 4.0f*exp(-V/18.0f);
}
float alpha_h(float V){
    return 0.07f*exp(-V/20.f);
}
float beta_h(float V){
    return 1.0f/(1.0f + exp((30.0f-V)/10.0f));
}

__kernel void update_params(__global float* V, __global float* M, __global float* H, __global float* N,
                            __global float* V_next, __global float* M_next, __global float* H_next, __global float* N_next,
                            __global float* change_rate, __global const int* sizes){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int center = INDEX(pos);
    int left = INDEX((pos + (int2) (1, 0)));
    int up = INDEX((pos + (int2) (0, 1)));
    int right = INDEX((pos - (int2) (1, 0)));
    int down = INDEX((pos - (int2) (0, 1)));

    float v = V[center];
    float m = M[center];
    float h = H[center];
    float n = N[center];

    float INa = gNa*h*m*m*m * (v - ENa);
    float IK = gK*n*n*n*n*(v - EK);
    float Ileak = gL*(v - EL);


    float2 fp = (float2) ((float)pos.x, (float)pos.y) / ((float) sizes[0]);
    float2 D = (float2) (fp.x, fp.y);
    D = normalize(D);
    //D += 2.0f;
    D /= 400.0f;
    float ks = (sin(fp.x*40) + 1.0f)/2.0;
    float kc = 1.0f - ks;
    float ks1 = (sin(fp.y*40) + 1.0f)/2.0;
    float kc1 = 1.0f - ks1;
    //ks = 3.0f*ks;
    //kc = 3.0f*kc;
    float I = (change_rate[left ]*(V[left ] - V[center])*ks +
               change_rate[right]*(V[right] - V[center])*kc +
               change_rate[up   ]*(V[up   ] - V[center])*kc1 +
               change_rate[down ]*(V[down ] - V[center])*ks1)*conductivity;

    float dv_dt = (I-(INa + IK + Ileak)/C_coof);
    float dm_dt = (alpha_m(v)*(1-m) - beta_m(v)*m);
    float dh_dt = (alpha_h(v)*(1-h) - beta_h(v)*h);
    float dn_dt = (alpha_n(v)*(1-n) - beta_n(v)*n);

    V_next[center] = V[center] + change_rate[center]*dv_dt*dt;
    M_next[center] = M[center] + change_rate[center]*dm_dt*dt;
    H_next[center] = H[center] + change_rate[center]*dh_dt*dt;
    N_next[center] = N[center] + change_rate[center]*dn_dt*dt;
}

__kernel void clear(__global float* array, __global const float* value, __global const int* sizes){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    array[INDEX(pos)] = *value;
}

__kernel void get_image(__global float* V, __global float* M, __global float* H, __global float* N, __global float* change_rate, __global int* image, __global const int* sizes){
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    int ind = INDEX(pos);

    float chRateCol = clamp(change_rate[ind], 0.0f, 1.0f);

    float3 yellow = (float3)(1.0f, 1.0f, 0.0f);
    float3 blue = (float3)(0.0f, 0.0f, 1.0f);
    float3 color = (yellow*(clamp(V[ind] - min_image_V, .0f, max_image_V - min_image_V)/(max_image_V - min_image_V)) + blue*clamp(change_rate[ind], 0.0f, 1.0f))*255.0f*chRateCol;
    /*
    float Vcol = clamp(V[ind] - min_image_V, .0f, max_image_V - min_image_V)/(max_image_V - min_image_V);
    float Mcol = clamp(M[ind], 0.f, 1.f);
    float Hcol = clamp(H[ind], 0.f, 1.f);
    float Ncol = clamp(N[ind], 0.f, 1.f);
    float3 color = (float3)(Mcol, Hcol, 0.5f + Ncol*0.5f)*chRateCol*255.0f;*/
    image[ind*3 + 0] = (int)color.r;
    image[ind*3 + 1] = (int)color.g;
    image[ind*3 + 2] = (int)color.b;
}

__kernel void add_I(__global float* V, __global float* change_rate, __global const int* sizes, __global const int* position, __global const float* radius, __global float* value){
    int2 pos = (int2)(get_global_id(0) + position[0], get_global_id(1) + position[1]);
    
    float2 p = (float2)(get_global_id(0), get_global_id(1)) - *radius;

    int ind = INDEX(pos);
    float L = length(p);
    if (L <= radius[0]){
        V[ind] = (*value) * change_rate[ind];
    }
}

__kernel void set_change_coof(__global float* change_rate, __global const int* sizes, __global const int* position, __global const float* radius, __global float* value){
    int2 pos = (int2)(get_global_id(0) + position[0], get_global_id(1) + position[1]);
    
    float2 p = (float2)(get_global_id(0), get_global_id(1)) - *radius;
    int ind = INDEX(pos);
    float L = length(p);
    if (L <= radius[0]){
        change_rate[ind] = (*value);
    }
}
/*
__kernel void draw_particle( __global float* Posd, __global float* movement, __global const int* world_sizes, __global const int* position, __global const float* radius, __global float* value){
    int2 pos = (int2)(get_global_id(0) + position[0], get_global_id(1) + position[1]);
    pos.x = min(max(pos.x, 0), world_sizes[0]);
    pos.y = min(max(pos.y, 0), world_sizes[1]);
    float2 p = (float2)(get_global_id(0), get_global_id(1)) - *radius;
    int ind = pos.x*world_sizes[1] + pos.y;
    float L = length(p);
    if (L <= radius[0]){
        Posd[ind] += value[0]*(sin(10.0*(p.x)))*(1.0f - L/radius[0]);
    }
}*/