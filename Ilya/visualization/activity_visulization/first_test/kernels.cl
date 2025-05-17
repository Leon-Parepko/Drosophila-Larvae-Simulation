#define get_activity(t, i) activity[*t*activity_size[0] + i]
#define W(i, j) w[i*activity_size[0] + j]

#ifndef PUSHING_FORCE
#define PUSHING_FORCE 1.0f
#endif
#ifndef PULLING_FORCE
#define PULLING_FORCE 1.0f
#endif

float pushing_function(float w){
    return PUSHING_FORCE * 1.0f/(1.0f + w*w);
}

float pulling_function(float m){
    return PULLING_FORCE * (1.0f - 1.0f/(1.0f + m*m));
}

__kernel void simulate_clasters(
    __global const float *activity,
 __global const float *w, __global float *positions,
  __global float *next_positions, __global const int* activity_size,
   __global const int *t, __global const float* _dt){
    int ind = get_global_id(0);
    const float dt = *_dt;

    float2 direction = (float2) (0.f, 0.f);
    float2 speed = (float2) (0.f, 0.f);

    for (int i = 0; i < *activity_size; i++){
        if (i != ind){
            direction.x = positions[ind*2 + 0] - positions[i*2 + 0];
            direction.y = positions[ind*2 + 1] - positions[i*2 + 1];
            float r = length(direction);
            r += 0.1f; // Avoid division by zero
            direction = normalize(direction);

            speed += direction*(pushing_function(W(ind, i))/r - pulling_function(W(ind, i) * (get_activity(t, i) - get_activity(t - 1, i))));
            
        }
    }
    next_positions[ind*2 + 0] = positions[ind*2 + 0] + speed.x*dt;
    next_positions[ind*2 + 1] = positions[ind*2 + 1] + speed.y*dt;
}