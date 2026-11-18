// Freestanding 4 x 4 signed i32 C += A * B check for AME_gem5.
// The actual result comes exclusively from the MLIR-generated AME kernel.
// Scalar multiplication below computes an independent expected result only.

enum { Side = 4, GuardWords = 16, Guard = 0x12345678 };

_Static_assert(sizeof(int) == 4, "the AME fixture requires 32-bit integers");

extern int ame_run(const int *a, const int *b, int *c);

_Alignas(64) static int a[Side][Side] = {
    {1, -2, 3, 4}, {-5, 6, 2, -1}, {7, 0, -3, 2}, {4, -6, 1, 5}};
_Alignas(64) static int b[Side][Side] = {
    {2, 3, -1, 4}, {-2, 5, 6, 1}, {7, -4, 2, -3}, {1, 2, -5, 6}};

static const int originalA[Side][Side] = {
    {1, -2, 3, 4}, {-5, 6, 2, -1}, {7, 0, -3, 2}, {4, -6, 1, 5}};
static const int originalB[Side][Side] = {
    {2, 3, -1, 4}, {-2, 5, 6, 1}, {7, -4, 2, -3}, {1, 2, -5, 6}};
static const int initialC[Side][Side] = {
    {3, -1, 5, 2}, {-4, 7, 1, -2}, {6, 3, -5, 4}, {-3, 2, 8, -6}};

// A complete cache line of guards keeps the output itself 64-byte aligned.
_Alignas(64) static struct {
  int before[GuardWords];
  int values[Side][Side];
  int after[GuardWords];
} output = {
    .values = {{3, -1, 5, 2}, {-4, 7, 1, -2}, {6, 3, -5, 4}, {-3, 2, 8, -6}}};

int main(void) {
  for (int i = 0; i < GuardWords; ++i)
    output.before[i] = output.after[i] = Guard;

  int status = ame_run(&a[0][0], &b[0][0], &output.values[0][0]);
  if (status)
    return status;

  for (int i = 0; i < GuardWords; ++i)
    if (output.before[i] != Guard || output.after[i] != Guard)
      return 2;

  for (int row = 0; row < Side; ++row)
    for (int col = 0; col < Side; ++col) {
      if (a[row][col] != originalA[row][col] ||
          b[row][col] != originalB[row][col])
        return 3;
      int expected = initialC[row][col];
      for (int k = 0; k < Side; ++k)
        expected += originalA[row][k] * originalB[k][col];
      if (output.values[row][col] != expected)
        return 1;
    }

  return 0;
}
