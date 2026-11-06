// RUN: buddy-opt %s --lower-bosc-ame --split-input-file --verify-diagnostics
//
// Reject forged or mistyped CPU DDR policy helpers: reserved names must be
// external () -> () funcs, and policy provenance on calls must be unit-valued.

module attributes {bosc_ame.target = "nr-fpga", bosc_ame.cpu_ddr_policy} {
  // expected-error @+1 {{reserved BOSCAME DDR policy helper @bosc_ame_ddr_read_begin must be an external func.func with type () -> ()}}
  func.func @bosc_ame_ddr_read_begin() {
    return
  }
}

// -----
module attributes {bosc_ame.target = "nr-fpga", bosc_ame.cpu_ddr_policy} {
  // expected-error @+1 {{reserved BOSCAME DDR policy helper @bosc_ame_ddr_write_begin must be an external func.func with type () -> ()}}
  func.func private @bosc_ame_ddr_write_begin(i32)
}

// -----
module attributes {bosc_ame.target = "nr-fpga", bosc_ame.cpu_ddr_policy} {
  func.func private @unrelated()
  func.func @forged_provenance() {
    // expected-error @+1 {{compiler DDR policy transition requires a reserved external () -> () helper and unit policy provenance}}
    func.call @unrelated() {bosc_ame.cpu_ddr_policy} : () -> ()
    return
  }
}

// -----
module attributes {bosc_ame.target = "nr-fpga", bosc_ame.cpu_ddr_policy} {
  func.func private @bosc_ame_ddr_read_begin()
  func.func @malformed_provenance() {
    // expected-error @+1 {{compiler DDR policy transition requires a reserved external () -> () helper and unit policy provenance}}
    func.call @bosc_ame_ddr_read_begin() {bosc_ame.cpu_ddr_policy = 1 : i32} : () -> ()
    return
  }
}
