#include <mujoco/mujoco.h>
#include <stdio.h>
#include <stdlib.h>

static void print_usage(const char* prog) {
  printf("Usage: %s [xml] [inner_iters] [activation] [qvel] [len0] [len_step] [count]\n",
         prog);
}

int main(int argc, char** argv) {
  const char* xml = (argc > 1) ? argv[1] : "../model_compliant_muscle_test.xml";
  int inner = (argc > 2) ? atoi(argv[2]) : 1;
  double activation = (argc > 3) ? atof(argv[3]) : 1.0;
  double qvel = (argc > 4) ? atof(argv[4]) : 0.0;
  double len0 = (argc > 5) ? atof(argv[5]) : -0.1472;
  double len_step = (argc > 6) ? atof(argv[6]) : 0.003;
  int count = (argc > 7) ? atoi(argv[7]) : 100;

  char load_error[1024] = {0};
  mjModel* m = mj_loadXML(xml, NULL, load_error, sizeof(load_error));
  if (!m) {
    printf("Could not load model: %s\n", load_error);
    print_usage(argv[0]);
    return 1;
  }

  mjData* d = mj_makeData(m);
  if (!d) {
    printf("Could not allocate data\n");
    mj_deleteModel(m);
    return 1;
  }

  m->opt.timestep = 1.0 / 1200.0;
  mj_resetData(m, d);

  printf("# len  ten_len  act_force  l_ce  v_ce  l_se\n");
  for (int i = 0; i < count; i++) {
    double length = len0 + len_step * i;
    // if (length < 0.001) length = 0.001;

    for (int k = 0; k < inner; k++) {
      d->qvel[0] = qvel;
      d->act[0] = activation;
      d->ctrl[0] = activation;
      d->qpos[0] = -length;
      mj_forward(m, d);
    }

    double ten_len = d->ten_length[0];
    double force = d->actuator_force[0];
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n",
           length, ten_len, force,
           d->muscle_l_ce[0], d->muscle_v_ce[0], d->muscle_l_se[0]);
  }

  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}

