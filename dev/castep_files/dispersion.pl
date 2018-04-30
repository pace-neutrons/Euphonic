#! /usr/bin/env perl
#
# Parse a "<>.castep" or "<>.phonon" or "<>.bands" output file from
# New CASTEP for vibrational frequency data and output an xmgrace plot
# of the electronic or vibrational band structure or dispersion.
#
#**********************************************************************
#  This program is copyright (c) Keith Refson. 1999-2012
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#**********************************************************************

use POSIX;
use Getopt::Long;
#use Math::Complex;
use strict;
#use warnings;

sub usage {
   printf STDERR "Usage: dispersion.pl [-xg] [-ps|-eps] [-np] <seed>.castep|<seed>.phonon ... \n";
   printf STDERR "       Extract phonon or bandstructure data from .castep, .phonon or .bands files\n";
   printf STDERR "       and optionally prepare a bandstructure/dispersion plot using XMGRACE as a backend.\n";
   printf STDERR "    -xg           Write a script and invoke GRACE to plot data.\n";
   printf STDERR "    -gp           Write a script and invoke GNUPLOT to plot data.\n";
   printf STDERR "    -mono         Create monochrome xmgrace plot\n";
   printf STDERR "    -ps           Invoke GRACE to plot data and write as a PostScript file.\n";
   printf STDERR "    -eps          Invoke GRACE to plot data and write as an encapsulated.\n                  PostScript (EPS) file.\n";
   printf STDERR "    -np           Do not plot data, write a GRACE script.\n";
   printf STDERR "    -nj           Do not attempt to perform eigenvector matching on phonon data for joined dispersion plots\n";
   printf STDERR "    -bs           Read band-structure from <>.castep or <>.bands.\n";
   printf STDERR "    -up           Extract and plot only spin up from <>.castep or <>.bands (implies -bs).\n";
   printf STDERR "    -down         Extract and plot only spin down from <>.castep or <>.bands (implies -bs).\n";
   printf STDERR "    -symmetry SYM Label plot according to Brillouin Zone of SYM=sc/fcc/bcc/hexagonal/tetragonal/orthorhombic....\n";
   printf STDERR "    -expt FILE    Read experimental data from EXPT and overplot.\n";
   printf STDERR "    -dat          Reread standard output from previous run and plot.\n";
   printf STDERR "    -ftol f       Set maximum discrepancy tolerance for phonon branch joining.\n";
   printf STDERR "    -sf s         Multiply frequencies by scaling factor of s.\n";
   printf STDERR "    -units s      Convert output to specified units for plotting.\n";
   printf STDERR "    -v            Be verbose about progres\n";
die;
}

my $number  = qr/-?(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee][+-]?\d{1-3})?/o;
my $fnumber = qr/-?(?:\d+\.?\d*|\d*\.?\d+)/o;
#my $fnumber = qr/-?(?=\d|\.\d)\d*(\.\d*)?/o;

my $qtol = 5.0e-6;
my $freq_close_tol = -1.0;
my $pi=3.14159265358979;

my $castep_phonon_re = qr/^ +\+ +q-pt= +\d+ \( *( *$fnumber) *($fnumber) *($fnumber)\) +($fnumber) +\+/o;
my $dot_phonon_re    = '';

my $castep_bs_re = qr/^  \+ +Spin=(\d) kpt= +(\d+) \( *( *$fnumber) *($fnumber) *($fnumber)\) +kpt-group= +\d +\+/o;
my $dot_bands_re    = '';

my $castep_re;

my ($grflag, $gpflag, $plotflag, $datflag,  $psflag, $qlabels, $symmetry, $exptfile, $exptq, $abscissa,$verbose) = ("", "", "","","","","","","","");
my ($title, $fileroot, $units, $unit_label, $unit_conv, $input_unit, $fermi_u, $fermi_d, $i,$freq,$exptn, $nspins, $colour, $mono, $scalefactor, $is_dot_castep);

my (@freqs, @qpts, @weights, @cell, @recip, @abscissa);

my ($opt_xg, $opt_gp, $opt_mono, $opt_np, $opt_ps, $opt_eps, $opt_bs,  $opt_up, $opt_down, $opt_dat, $opt_nj, $opt_sym, $opt_expt, $opt_ftol,$opt_sf,$opt_units,$opt_help,$opt_v) = (0,0,0,0,0,0,"",0,0,0,"","","","","","","");

my ($alat, $blat, $clat, $alpha, $beta, $gamma);

&GetOptions("xg"   => \$opt_xg, 
	    "gp"   => \$opt_gp, 
	    "mono" => \$opt_mono, 
	    "np"   => \$opt_np, 
	    "nj"   => \$opt_nj, 
	    "ps"   => \$opt_ps, 
	    "eps"  => \$opt_eps, 
	    "bs"   => \$opt_bs, 
	    "up"   => \$opt_up, 
	    "down" => \$opt_down, 
	    "dat"  => \$opt_dat, 
	    "symmetry=s" => \$opt_sym,
	    "expt=s" => \$opt_expt,
	    "ftol=f" => \$opt_ftol,
	    "sf=f"   => \$opt_sf,
	    "units=s"   => \$opt_units,
	    "h|help"    => \$opt_help,
	    "v"    => \$opt_v )|| &usage();

if ($opt_help || $opt_ps && $opt_eps) {&usage()};

$nspins = 1;
$grflag= $opt_xg if $opt_xg;
$gpflag= $opt_gp if $opt_gp;
$mono = $opt_mono;
$plotflag = 1;
$plotflag= 0 if $opt_np;
$psflag = 0;
$psflag= 1 if $opt_ps;
$psflag= 2 if $opt_eps;
$grflag = 1 if $psflag && ! ($grflag || $gpflag) ;
$exptfile = "";
$exptfile = $opt_expt if $opt_expt;
$datflag = 1 if $opt_dat;
$freq_close_tol = $opt_ftol if $opt_ftol;
$verbose = 0;
$verbose = 1 if $opt_v;
$opt_bs = 1 if( $#ARGV >= 0 && $ARGV[0]=~ /\.bands$/ );
$opt_bs = 1 if( $opt_up | $opt_down);
$opt_up = $opt_down = 1  if( $opt_bs && !($opt_up || $opt_down));
$opt_bs = 0 if( $#ARGV >= 0 && $ARGV[0]=~ /\.phonon$/ );
$scalefactor = 1.0;
$scalefactor = $opt_sf if $opt_sf;

$symmetry="";
$symmetry = $opt_sym if $opt_sym;

if ($#ARGV >= 0) {
  $title=$ARGV[0];
  $title=~s/\.(phonon|castep|bands)//;
} else {
  &usage();
}
$fileroot=$title;

#
# Set up units for output from options and defaults
#
$units = $opt_units;
$unit_label = "default";
$unit_conv = 1.0;
my %unit_scale = (
    # Energies and frequencies - relative to default eV
    "mev"  => 1000.0,
    "thz"  => 241.7989348,
    "cm-1" => 8065.73,
    "ev"   => 1.0,
    "ha"   => 1/27.21138505,
    "mha"  => 1/0.02721138505,
    "ry"   => 1/13.605692525,
    "mry"  => 1/0.013605692525
    );

if( $opt_bs ) {
  $units = "eV" if( $opt_units eq "" );
  $input_unit = 1.0;
  $castep_re = $castep_bs_re;
} else {
  $units = "cm-1" if( $opt_units eq "" );
  $input_unit = 8065.73;
  $castep_re = $castep_phonon_re;
}

$unit_conv = $unit_scale{lc($units)} / $input_unit;

if( $grflag ) {
  for( $units ) {
    if ( /eV/i )   { $unit_label = '\\f{Symbol}e\\f{} (eV)';}
    if ( /Ry/i )   { $unit_label = '\\f{Symbol}e\\f{} (Ry)';}
    if ( /mRy/i )  { $unit_label = '\\f{Symbol}e\\f{} (mRy)';}
    if ( /Ha/i )   { $unit_label = '\\f{Symbol}e\\f{} (Ha)';}
    if ( /mHa/i )  { $unit_label = '\\f{Symbol}e\\f{} (mHa)';}
    if ( /THz/i )  { $unit_label = '\\f{Symbol}w\\f{} (THz)';}
    if ( /cm-1/i ) { $unit_label = '\\f{Symbol}w\\f{} (cm\\S-1\\N)';}
    if ( /meV/i )  { $unit_label = '\\f{Symbol}w\\f{} (meV)';}
  }
} elsif ($gpflag ) {
  for( $units ) {
    if ( /eV/i )   { $unit_label = '{/Symbol e} (eV)';}
    if ( /Ry/i )   { $unit_label = '{/Symbol e} (Ry)';}
    if ( /mRy/i )  { $unit_label = '{/Symbol e} (mRy)';}
    if ( /Ha/i )   { $unit_label = '{/Symbol e} (Ha)';}
    if ( /mHa/i )  { $unit_label = '{/Symbol e} (mHa)';}
    if ( /THz/i )  { $unit_label = '{/Symbol w} (THz)';}
    if ( /cm-1/i ) { $unit_label = '{/Symbol w} (cm^{-1})';}
    if ( /meV/i )  { $unit_label = '{/Symbol w} (meV)';}
  }
}

if ($plotflag ) {
  if( $grflag) {
    if( $psflag ) {
      open STDOUT, "| gracebat -pipe -nosafe" or die "Failed to open pipe to GRACEBAT process";
    } else {
      open STDOUT, "| xmgrace -pipe" or die "Failed to open pipe to GRACE process";
    }
  } elsif ($gpflag) {
    if( $psflag ) {
      open STDOUT, "| gnuplot -persist - 2>/dev/null 1>&2" or die "Failed to open pipe to GNUPLOT process";
    } else {
      open STDOUT, "| gnuplot -persist - 2>/dev/null 1>&2" or die "Failed to open pipe to GNUPLOT process";
    }
  }
}

#$pos = tell ARGV;
$_ = <>;
seek ARGV,0,0;

$i = -3;
do {
    $_ = <>;

    $is_dot_castep = 1 if ( /^ \+-+\+\s*$/ );
    $i++;
} while ($i <= 0 and ! $is_dot_castep);

seek ARGV,0,0;

if ($datflag) {
  reread_data(\@freqs,\@qpts,\@abscissa);
  $abscissa = \@abscissa;
} else {
  if ( $is_dot_castep ) {
    if(  $opt_bs ) {
      read_dot_castep_bands(\@freqs,\@qpts, \$fermi_u, \$fermi_d,\@cell);
    } else {
      read_dot_castep_phonon(\@freqs,\@qpts, \@cell);
    }
  } else {
    if(  $opt_bs ) {
      read_dot_bands(\@freqs,\@qpts, \$fermi_u, \$fermi_d,\@weights, \@cell);
    } else {
      read_dot_phonon(\@freqs,\@qpts, \@cell);
    }
  }

  &MATinv3x3(\@cell, \@recip);

  if( $verbose ) {
    for $i (0..2) {
      printf STDERR "CELL[%d] = %f %f %f    %f %f %f\n", $i,$cell[$i][0],$cell[$i][1],$cell[$i][2],
	$recip[$i][0],$recip[$i][1],$recip[$i][2];
    }
  }
  #
  # Compute abscissa for plot
  #
  $abscissa = make_abscissa(\@qpts,\@recip);
}
#
if ( $#qpts < 0 or $#freqs < 0 ) {
  if(  $opt_bs ) {
    die "Failed to read any bandstructure eigenvalues - check file and arguments\n";
  } else {
    die "Failed to read any phonon frequencies - check file and arguments\n";
  }
}

# Convert to units specified
for $freq (@freqs) {
  map {$_ *= $unit_conv}  @{$freq};
}
# Apply scaling factor
for $freq (@freqs) {
  map {$_ *= $scalefactor}  @{$freq};
}
#
# Prepend to frequencies list to get [x,f1,f2,f3,...]
#
$i = 0;
for $freq (@freqs) {
  unshift @{$freq},$$abscissa[$i++];
}

#for $freq (@freqs) {
#  for $branch (@$freq){
#    print STDERR $branch,"  ",;
#  }
#  print STDERR "\n";
#}

#
# Special handling for hexagonal with 60 degree axes
# Make conditional so as not introduce dependency on cell parameters which breaks data reread.
#
if ( $#cell > 0 ) {
    ($alat, $blat, $clat, $alpha, $beta, $gamma) = MATtoABC(\@cell);
    $symmetry = "hexagonal60" if ( fabs($gamma - 60.0) < 1.0e-4 and $symmetry eq "hexagonal" );
}

$qlabels = make_qlabels(\@qpts, $abscissa);

#for $qlabel (@$qlabels) {
#  print STDERR $$qlabel[0]," ",$$qlabel[1]," ",$$qlabel[2]," ",$$qlabel[3],"\n";
#}

$exptq = read_exptl($exptfile) if ( $exptfile ne "" );

$exptn = map_exptl($exptq, $qlabels) if ( $exptfile ne "" );

#print STDERR "Number of experimental datasets = ",$#{$exptq}+1,"\n";
#$iset = 1;
#for $eset (@$exptq){
#  print STDERR "Set ",$iset,": Number of data = ",$#{$eset}+1,"\n";
#  $iset++;
#}

#for $qlabel (@qlabels) {
#  printf STDOUT "%d  %f %f %f\n",$$qlabel[0],$$qlabel[1],$$qlabel[2],$$qlabel[3];
#}


if( $grflag ) {
  if(  $opt_bs ) {
    graceout($qlabels, \@freqs, $fileroot, $psflag, $unit_label, $title,$fermi_u);
  } else {
    graceout($qlabels, \@freqs, $fileroot, $psflag, $unit_label, $title,"" ,$exptn);
  }
} elsif( $gpflag ) {
  if(  $opt_bs ) {
    gnuplotout($qlabels, \@freqs, $fileroot, $psflag, $unit_label, $title,$fermi_u);
  } else {
    gnuplotout($qlabels, \@freqs, $fileroot, $psflag, $unit_label, $title,"" ,$exptn);
  }

} else {
  write_data(*STDOUT, \@freqs, \@qpts, 1);
}

#
# End of executable part -- subroutines follow.
#

sub reread_data {
  my($freqs, $qpts) = @_;
  my( $count, $pt);
  my(@freqk,@q,@line, $nfreqs, $pltcoord );

  $count = 0;

  while ( <> ) {
    #
    # <>.txt
    #
    print STDERR $_ if $verbose;
    @line = split;
    
    $pt = shift @line;
    @q = splice @line, 0, 3;
    $pltcoord = shift @line;
    push @abscissa,$pltcoord;
    @freqk = @line;

    die "Need sequentially numbered lines ($pt != $count)" unless $pt == $count;
    if( $count > 1 ) {
      die "Every line should have same number of values" unless $nfreqs == $#freqk;
    }
    
    $count++;
    $nfreqs = $#freqk;
    push @$freqs, [@freqk];
    push @$qpts, [@q];
    
  }
}

sub read_dot_castep_phonon {
  my($freqs, $qpts,$cell) = @_;
  my(@freqk,@q,@vv, $junk);
  my($freq_unit) = ("cm-1");

  while ( <>) {
    #
    # <>.castep
    #
    if(/^\s+Real Lattice[(]A[)]\s+Reciprocal Lattice[(]1\/A[)]/ .. /^\s+Lattice parameters[(]A[)]/ ) {
      if (/^(\s*$number){6}\s*$/ ){
	($vv[0],$vv[1],$vv[2],$junk) = split; push @$cell, [@vv];
	$_=<>; ($vv[0],$vv[1],$vv[2],$junk) = split; push @$cell, [@vv];
	$_=<>; ($vv[0],$vv[1],$vv[2],$junk) = split; push @$cell, [@vv];
      }
    }
    if( /^ output\s+frequency unit\s+: *([^\s]+)/ ) {
	$freq_unit = $1;
    }
    if (/$castep_phonon_re/o) {
      @q=($1,$2,$3);
      print STDERR "Reading ",$q[0]," ",$q[1]," ",$q[2],"\n" if $verbose;
      #print "   ";
      last if eof ARGV;
      $_ = <>;
      last if eof ARGV;
      $_ = <> if /^ +\+ +Effective cut-off =/;
      last if eof ARGV;
      $_ = <> if /^ +\+ +q->0 along \( *( *$fnumber) *($fnumber) *($fnumber)\) +\+/;
      last if eof ARGV;
      $_ = <> if /^ +\+ -+ \+/;
      @freqk=();

      while (<>) {
	if (/^ +\+ +\d+ +($fnumber)( +\w)?(( +$fnumber)?( +\w)?){2} *\+/) { 
	  push @freqk, $1;
#	} elsif (/^  \+ +q->0 along \( *( *$fnumber) *($fnumber) *($fnumber)\) +\+/ ) {
	} elsif (/^ +\+ -+ \+/) {
	  last;
	} elsif (/^ +\+ +.*\+/) {
        } else {
	  last;
	}
      }
      print STDERR "Frequencies in units of $freq_unit\n" if $verbose;
      print STDERR @freqk,"\n" if $verbose;
      for $freq (@freqk) {
	  $freq *= $unit_scale{'cm-1'}/$unit_scale{$freq_unit};
      }
      push @$freqs, [@freqk];
      push @$qpts, [@q];
      last if eof ARGV;
    }
  }
}

sub read_dot_castep_bands {
  my($freqs, $qpts, $fermi_u, $fermi_d,$cell) = @_;
  my(@freqk,@q,@kp_map,@vv,$junk, $kp_count,$spin,$kpt);
  my($energy_unit) = ("eV");
  
  $kp_count = 0;
  while ( <>) {
    #
    # <>.castep
    #
    if(/^\s+Real Lattice[(]A[)]\s+Reciprocal Lattice[(]1\/A[)]/ .. /^\s+Lattice parameters[(]A[)]/ ) {
      if (/^(\s*$number){6}\s*$/ ){
	($vv[0],$vv[1],$vv[2],$junk) = split; push @$cell, [@vv];
	$_=<>; ($vv[0],$vv[1],$vv[2],$junk) = split; push @$cell, [@vv];
	$_=<>; ($vv[0],$vv[1],$vv[2],$junk) = split; push @$cell, [@vv];
      }
    }
    if( /^ output\s+energy unit\s+: *([^\s]+)/ ) {
	$energy_unit = $1;
    }
    #
    $$fermi_u = $1 if /^  \+  Fermi energy for spin  up  electrons is:\s+ ($number)[^+]+\+/;
    $$fermi_d = $1 if /^  \+  Fermi energy for spin down electrons is:\s+ ($number)[^+]+\+/;
    if (/$castep_bs_re/o) {
      ($spin,$kpt,@q)=($1,$2,$3,$4,$5);
      print STDERR "Reading Spin=$spin;kpt=$kpt  ($q[0],$q[1],$q[2])\n" if $verbose;
      #print "   ";
      last if eof ARGV;
      $_ = <>;
      last if eof ARGV;
      $_ = <> if /^  \+ -+ \+/;
      @freqk=();

      while (<>) {
	if (/^  \+ +\d+ +($fnumber)( +\*)?( +$fnumber)* *\+/) { 
	  push @freqk, $1;
#	} elsif (/^  \+ +q->0 along \( *( *$fnumber) *($fnumber) *($fnumber)\) +\+/ ) {
	} elsif (/^  \+ -+ \+/) {
	  last;
	} elsif (/^  \+ +.*\+/) {
	} else {
	  last;
	}
      }
      print STDERR "Energies in units of $energy_unit\n" if $verbose;
      print STDERR @freqk,"\n" if $verbose;
      for $freq (@freqk) {
	  $freq *= $unit_scale{'eV'}/$unit_scale{$energy_unit};
      }
      #
      # Keep track of kpt ordering for spin=2.  Assumes all spin=1 precede spin=2.
      #
      if( $spin == 1 ) {
	$kp_map[$kpt] = $kp_count++;
	push @$freqs, [@freqk];
	push @$qpts, [@q];
      } else {
	push @{$$freqs[$kp_map[$kpt]]}, @freqk;
	$nspins++ if $nspins < 2;
      }
      last if eof ARGV;
    }
  }
}

sub min {
  my ($a,$b) = @_;
  return $a if $a < $b;
  return $b;
}

sub max {
  my ($a,$b) = @_;
  return $a if $a > $b;
  return $b;
}

sub read_dot_phonon {
  my($freqs, $qpts, $cell) = @_;
  my(@freqk,@freqk_r,@freqk_o,@q,@eigvec, @oeigvec, $qmapt, @qmap);
  my(@coords,@xyz,@corrmat,@vv,@qturn);
  my($x,$y,$z,$nions,$nmodes, $ion,$aty, $mass, $count, $mode, $omode,$dot);
  my($qmode,$qomode,$dot_same,$savedot,$maxdot,$save,$qmaxdot, $f);

  #
  # Read frequencies from <>.phonon file
  #
  # Also reads eigenvectors and attempts to join up branches
  # on the dispersion curve using a heuristic algorithm based
  # assuming close to orthonormality of eigenvectors at
  # adjacent q-points.
  #
  while (<>) {
    #
    # <>.phonon
    #
    # Read header
    $nions = $1 if /^ Number of ions\s*(\d+)/;
    $nmodes = $1 if /^ Number of branches\s*(\d+)/;
    if( /^ Unit cell vectors/ ) {
      $_ = <>; @vv = split; push @$cell, [@vv];
      $_ = <>; @vv = split; push @$cell, [@vv]; 
      $_ = <>; @vv = split; push @$cell, [@vv];
    }
    if( / Fractional Co-ordinates/../ end.*header/i) {
      if( /^\s+\d+(\s+$fnumber){3}\s+[A-Za-z]{1,2}\s+$fnumber/ ) {
	($ion,$x,$y,$z,$aty,$mass) = split;
	push @coords, [$x,$y,$z];
      }
    }
    #
    # Initialize q-point remapping array
    #
    if( / end .*header/i ) {
      @qmap = 0..$nmodes-1;
    }
    #
    # Found start of block of frequencies
    #
    if (/^ +q-pt= +\d+ +($fnumber) +($fnumber) +($fnumber)( +$fnumber){0,4}/) {
#    if (/^ +q-pt= +\d+ +( *$fnumber) *($fnumber) *($fnumber)( *$fnumber){0,4}$/) {
      @q=($1,$2,$3);
      #print STDERR " ",$q[0]," ",$q[1]," ",$q[2];
      #print "   ";
      @freqk_r=();

      while (<>) {
	if (/^ +\d+ +($fnumber) */) {
	  push @freqk_r, $1;
	} else {
	  last;
	}
      }
#      print STDERR @freqk_r,"\n";
      push @$qpts, [@q];
      #
      # Find max frequency and tolerance if not set.
      #
      if ( $freq_close_tol < 0) {
	for $f (@freqk_r) {
	  $freq_close_tol = $f if $freq_close_tol < $f;
	}
	# 1/8 of max should be fine
	$freq_close_tol *= 0.125;
	$freq_close_tol = 50.0 if( $freq_close_tol < 0 );
      }
    }
    #
    # Found start of eigenvectors block
    #
    if (/^\s+Phonon Eigenvectors/) {
      $_ = <>;
      @eigvec = ();
      #
      # Read in set of eigenvectors
      #
      for $count ( 0..$nions*$nmodes-1 ) {
	($mode, $ion, @xyz) = split " ",<>;
	push @{$eigvec[$mode-1]}, @xyz;
      }
      if( ! $opt_nj ) {
	#
	# Remapping and eigenvector matching to join branches
	#
	$qmapt = [0..$nmodes-1];
	if ( $#{$qpts} > 0 && ! direction_changed(\@qturn,@q)) {
	  #	print STDERR "\n";
	  #
	  # Loop over current and previous modes, comparing eigenvectors
	  # and finding greatest exp(-iqr)-weighted dot product.
	  #
	  for $mode (0..$nmodes-1) {
	    for $omode (0..$nmodes-1) {
	      #	    if(    abs($freqk_r[$mode] - $freqk_r[$omode]) <  $freq_close_tol) {
	      $dot = abs(dot_eigens(\@coords,$eigvec[$mode], $oeigvec[$omode]));
	      #	    } else {
	      #	      $dot = -1;
	      #	    }
	      $corrmat[$mode][$omode] = $dot;
	      printf STDERR "%8.3g ",$corrmat[$mode][$omode] if $verbose > 1;
	    }
	    print STDERR "\n" if $verbose > 1;
	  }
	  
	  $qmapt = make_map(\@corrmat);
	  if( $#{$qmapt} == 0) {
	    for $mode (0..$nmodes-1) {
	      for $omode (0..$nmodes-1) {
		printf STDERR "%8.3g ",$corrmat[$mode][$omode];
	      }
	      print STDERR "\n";
	    }
	    die "Make_map returned error $$qmapt[0]";
	  }
	}
	#
	# Debugging output - why was crossing/not crossing decision taken?
	if( $verbose ) {
	  if( ! compare_arrays( $qmapt , [0..$nmodes-1])) {
	    print STDERR "Modes re-ordered at q-pt  $#{$qpts} = ($q[0],$q[1],$q[2])\n";
	    for $mode (0..$nmodes-1) {
	      printf STDERR "%3d",$$qmapt[$mode];
	    }
	    printf STDERR "\n";
	    for $mode (0..$nmodes-1) {
	      for $omode (0..$nmodes-1) {
		printf STDERR "%8.3g ",$corrmat[$mode][$omode];
	      }
	      print STDERR "\n";
	    }
	    print STDERR "\n";
	  } elsif( nrowgt(\@corrmat,0.5) > 1 ) {
	    print STDERR "Modes NOT re-ordered at q-pt  $#{$qpts} = ($q[0],$q[1],$q[2])\n";
	    for $mode (0..$nmodes-1) {
	      for $omode (0..$nmodes-1) {
		printf STDERR "%8.3g ",$corrmat[$mode][$omode];
	      }
	      print STDERR "\n";
	    }
	    print STDERR "\n";
	  }
	}
	#
	# Multiply into total mapping vector @qmap
	#
	@qmap = map {$qmap[$_]} @$qmapt;
	#
	# Reorder frequencies and store current set of eigenvectors
	#
	@freqk_o = @freqk_r;
	for $mode (0..$nmodes-1) {
	  $freqk[$qmap[$mode]] = $freqk_r[$mode];
	  $oeigvec[$mode] = [@{$eigvec[$mode]}];
	}
	push @$freqs, [@freqk];
      } else {
	push @$freqs, [@freqk_r];
      }
    }
  }
}

sub compare_arrays {
    my ($first, $second) = @_;
    no warnings;  # silence spurious -w undef complaints
    return 0 unless @$first == @$second;
    for (my $i = 0; $i < @$first; $i++) {
        return 0 if $first->[$i] ne $second->[$i];
    }
    return 1;
}

sub nrowgt {
  my($mat, $tol) = @_;
  my($i, $j, $count,$max);

  $max = 0;
  for $i (0..$#{$mat}) {
    $count = 0;
    for $j (0..$#{$mat}) {
      $count++ if $$mat[$i][$j] > $tol;
    }
    $max = $count if $max < $count;
  }
  $max;
}


sub make_map {
  my ($corrmat) = @_;
  my($i, $j, $n, $maxval,$maxpos, $err);
  my(@qmap, @doner, @donec, @count);

  $n = $#{$corrmat};

  for $i (0..$n) {
    $doner[$i] = 0;
    $donec[$i] = 0;
  }

  my (@corrh, @corrhs, $rec, $val,$k,$cnt);
  for $i (0..$n) {
    for $j (0..$n) {
      push @corrh, {val=>$$corrmat[$i][$j], i=>$i, j=>$j};
    }
  }

  @corrh = sort {$$b{val} <=> $$a{val}} @corrh;

  #Main loop over ordered values in $corrmat, largest first
  $cnt = 0;
  for $k (0..$n) {
    for $rec (@corrh) {
      $i = $rec->{i}; $j = $rec->{j};
      $val = $rec->{val};
      if( ! ( $doner[$i] || $donec[$j]) ) {
#	print STDERR "Make_Map: $cnt    $i  $j  $val\n";
	$qmap[$i] = $j;
	$doner[$i]++;
	$donec[$j]++;
	$cnt++;
	last;
      }
    }
  }
  
  for $i (0..$n) {
    $count[$qmap[$i]]++;
  }
  $err = 0;
  for $i (0..$n) {
     $err++ if( ! $doner[$i] || $count[$i] != 1);
  }  
  if( $err ) {
    for $i (0..$n) {print STDERR $doner[$i]," "};
    print STDERR "\n";
    for $i (0..$n) {print STDERR $qmap[$i]," "};
    print STDERR "\n";
    print STDERR "Failed to make correct mapping\n";
    return [(-1)];
  }
  if( $verbose ) {
    for $i (0..$n) {print STDERR $qmap[$i]," "};
    print STDERR "\n";
  }
  [@qmap];

}
sub read_dot_bands {
  my($freqs, $qpts, $fermi, $fermi1, $weights,$cell) = @_;
  my(@eigk,@k, @freq_list, @qpt_list, @nk_list, @wt_list, @vv);
  my($wt, $first, $Hartree,$nk,$ink,$ink_save, $ncount, $active);

  $Hartree =27.2114;
  $first = 1;
  $nk = -1;
  $ncount = 0;
  #
  # Pass 1 - read the data
  while (<>) {
    #
    # <>.bands
    #
    chop;
    if (/^K-point +(\d+) +($fnumber) *($fnumber) *($fnumber) *($fnumber)/) {
      $active = 0;
      if( ! $first ) {
	push @freq_list, [@eigk];
	push @qpt_list,  [@k];
	push @nk_list,   $nk;
	push @wt_list,   $wt;
	print STDERR "Read K-point $nk, ($qpt_list[$ncount-1][0],$qpt_list[$ncount-1][1],$qpt_list[$ncount-1][2])\n" if $verbose;
      }
      $first = 0;
      if( $1 == 1 and $ncount > 1 ) {
	if( $ncount == $nk+1) {
	  $nk = $ncount;
	} else {
	  print STDERR "Don't know what to do with duplicate k-point $ncount\n";
	  $nk = -1;
	}
      } else {
	$nk = $1-1;
      }
      @k=($2,$3,$4);
      $wt = $5;
      @eigk=();
      $ncount++;
    } elsif (/^Spin component 1/) {
      $active = $opt_up;
    } elsif (/^Spin component 2/) {
      $active = $opt_down;
      $nspins++ if ($nspins < 2);
    } elsif (/^ +($fnumber)\s*$/) {
      push @eigk, $1*$Hartree if $active;
    } elsif ( /Fermi energy \(in atomic units\) +($fnumber)/ ) {
      $$fermi = $1 * $Hartree;
    } elsif ( /Fermi energies \(in atomic units\) +($fnumber) +($fnumber)/ ) {
      $$fermi = $1 * $Hartree;
      $$fermi1 = $1 * $Hartree;
    } elsif( /^Unit cell vectors/ ) {
      $_ = <>; @vv = split; push @$cell, [@vv];
      $_ = <>; @vv = split; push @$cell, [@vv];
      $_ = <>; @vv = split; push @$cell, [@vv];
    }

  }
  push @freq_list, [@eigk];
  push @qpt_list,  [@k];
  push @nk_list,   $nk;
  push @wt_list,   $wt;
  print STDERR "Read K-point $nk, ($qpt_list[$ncount-1][0],$qpt_list[$ncount-1][1],$qpt_list[$ncount-1][2])\n" if $verbose;
  #
  # Reorder list and store into output arrays
  #
  $ink_save = -1;
  for $ink (0..$#nk_list) {
    $nk = $nk_list[$ink];
    print STDERR "Reordering K-point $nk, ($qpt_list[$ink][0],$qpt_list[$ink][1],$qpt_list[$ink][2])\n" if $verbose;
    if( $nk < 0 ) {
      $ink_save = $ink;
    } else {
      $$freqs[$nk] = $freq_list[$ink];
      $$qpts[$nk] =  $qpt_list[$ink];
      #$$weights[$nk] = $wt_list[$ink];
    }
  }
  #
  # Due to a CASTEP bug some state may be double-labelled. In that case we have a "spare" set
  # of frequencies in $ink_save.  See if we can guess where to put it.
  for $ink (1..$#$qpts) {
    if( $#{$$freqs[$ink]} < 1 ) {  # Aha!
    print STDERR "Placing missing K-point $ink, ($qpt_list[$ink_save][0],$qpt_list[$ink_save][1],$qpt_list[$ink_save][2])\n" if $verbose;
      
      $$freqs[$ink] = $freq_list[$ink_save];
      $$qpts[$ink] =  $qpt_list[$ink_save];
    }
  }

}

sub qlabel_string {
  my ($qpt) = @_;

  my ($gamma,$str,$val,$key, %labelhash);

  #
  # Normalise recip co-ordinates into [0,1) and scale to [0,24)
  #
  for $i ( 1..3 ) {
    $val = $$qpt[$i] - floor($$qpt[$i]);
    $val=sprintf "%.0f",24*$val;
    if ($i == 1) {
      $key = $val;
    }else {
      $key="${key},${val}";
    }
  }
#  print STDERR $key,"\n";
  $gamma = "\\f{Symbol}G\\f{}" if( $grflag );
  $gamma = "{/Symbol G}" if( $gpflag );

  if ( $symmetry eq "sc" or  $symmetry eq "cubic") {
    %labelhash = ( "0,0,0" => $gamma,
		   "12,0,0" => "X", "0,12,0" => "X", "0,0,12" => "X",
		   "12,12,0" => "M", "12,0,12" => "M", "0,12,12" => "M",
		   "12,12,12" => "R");
    $str = $labelhash{$key};
  } elsif  ( $symmetry eq "fcc" ) {
    %labelhash = ( "0,0,0" => $gamma,
		   "12,12,0"  => "X", "12,0,12"  => "X", "0,12,12"  => "X",
		   "12,18,6"  => "W", "18,6,12"  => "W", "6,12,18"  => "W",
		   "12,6,18"  => "W", "6,18,12"  => "W", "18,12,6"  => "W",
		   "9,15,0"   => "K", "9,0,15"   => "K", "0,9,15"   => "K",
		   "15,9,0"   => "K", "15,0,9"   => "K", "0,15,9"   => "K",
		   "15,15,6"  => "K", "15,6,15"  => "K", "6,15,15"  => "K",
		   "15,15,6"  => "K", "15,6,15"  => "K", "6,15,15"  => "K",
		   "9,9,18"   => "K", "9,18,9"   => "K", "18,9,9"   => "K",
		   "12,12,12" => "L");
    $str = $labelhash{$key};
  } elsif  ( $symmetry eq "fcc-afm" ) {
    %labelhash = ( "0,0,0" => $gamma,
		   "12,12,0"  => "X", "12,0,12"  => "X", "0,12,12"  => "X",
		   "12,0,0"   => "L", "0,12,0"   => "L", "0,0,12"   => "L",
		   "12,12,12" => "T");
    $str = $labelhash{$key};
  } elsif  ( $symmetry eq "bcc" ) {
    %labelhash = ( "0,0,0" => $gamma,
		   "18,18,18" => "P", "6,6,6" => "P",
		   "12,12,0"  => "N", "12,0,12" => "N", "0,12,12" => "N",
		   "12,0,0"   => "N", "0,0,12"  => "N", "12,0,12" => "N",
		   "12,12,12" => "H");
    $str = $labelhash{$key};
  } elsif ( $symmetry eq "tetragonal" ) {
    %labelhash = ( "0,0,0" => $gamma,
		   "12,0,0" => "X", "0,12,0" => "X", 
		   "0,0,12" => "Z",
		   "12,12,0" => "M", 
		   "12,0,12" => "R", "0,12,12" => "R",
		   "12,12,12" => "A");
    $str = $labelhash{$key};
  } elsif ( $symmetry eq "tetragonal-I" or $symmetry eq "b-tetragonal") {
    %labelhash = ( "0,0,0" => $gamma,
		   "12,12,0" => "X", "0,0,12"   => "X", 
		   "12,0,0"  => "N", "0,12,0"   => "N",
		   "12,0,12" => "N", "0,12,12"  => "N",
		   "6,6,6"   => "P", "18,18,18" => "P", 
#		   "7,7,17"  => "K.6", "17,17,7" => "K.9", 
		   "12,12,12" => "M");
    $str = $labelhash{$key};
  } elsif ( $symmetry eq "orthorhombic" ) {
    %labelhash = ( "0,0,0" => $gamma,
		   "12,0,0" => "X", "0,12,0" => "Y", "0,0,12" => "Z",
		   "12,12,0" => "S", "12,0,12" => "U", "0,12,12" => "T",
		   "12,12,12" => "R");
    $str = $labelhash{$key};
  } elsif  ( $symmetry eq "hexagonal" ) {
    %labelhash = ( "0,0,0" => $gamma,
		   "12,12,0"  => "M", "12,0,0"   => "M", "0,12,0"   => "M", 
		   "12,12,12" => "L", "12,0,12"  => "L", "0,12,12"  => "L", 
 		   "8,8,0"    => "K", "16,16,0"  => "K", 
 		   "8,8,12"   => "H", "16,16,12" => "H", 
		   "0,0,12" => "A");
    $str = $labelhash{$key};
  } elsif  ( $symmetry eq "hexagonal60" ) {
    %labelhash = ( "0,0,0" => $gamma,
		   "12,12,0"  => "M", "12,0,0"   => "M", "0,12,0"   => "M", 
		   "12,12,12" => "L", "12,0,12"  => "L", "0,12,12"  => "L", 
 		   "16,8,0"    => "K", "8,16,0"  => "K", 
 		   "16,8,12"   => "H", "8,16,12" => "H", 
		   "0,0,12" => "A");
    $str = $labelhash{$key};
  } elsif  ( $symmetry eq "trigonal" ) {
    %labelhash = ( "0,0,0" => $gamma,
		   "12,12,12" => "Z",
		   "12,0,0"   => "L", "0,12,0"   => "L", "0,0,12"   => "L", 
		   "12,12,0"  => "F", "0,12,12" => "F", "12,0,12"  => "F", 
		   );
    $str = $labelhash{$key};
  } elsif  ( $symmetry eq "trigonal-h" ) {
    %labelhash = ( "0,0,0" => $gamma,
		   "16,16,16" => "K",
		   "12,12,0"  => "M", "0,12,12" => "M", "12,0,12"  => "M", 
		   "12,12,12" => "A");
    $str = $labelhash{$key};
  } else {
    $str = $key;
    $str =~ s@\b18\b@3/4@g;
    $str =~ s@\b6\b@1/4@g;
    $str =~ s@\b12\b@1/2@g;
    $str =~ s@\b8\b@1/3@g;
    $str =~ s@\b16\b@2/3@g;
    $str =~ s@\b9\b@3/8@g;
    $str =~ s@\b15\b@5/8@g;
  }
  $str;
}

sub graceout {
    my($qlabels, $freqs, $plotfile, $psflag, $unit_label, $title, $fermi, $fexptl ) = @_;
    my($i, $onefile, $filenum, $nbranches, $nqpts, $nqexptl, $setnum, $symbol);
    my($maxqpts, $qpt, $nsets, $datum);

    $| = 1;
    $nbranches = $#{$freqs->[0]};
    $nqpts     = $#{$freqs};
    $maxqpts   = $$freqs[$nqpts][0];

    print   "\@timestamp def \"",scalar(localtime),"\"\n";
    print   "\@with g0\n";
    print   "\@title \"",$title,"\"\n";
    print   "\@view 0.150000, 0.250000, 0.700000, 0.850000\n";
    print   "\@world xmin 0\n";
    print   "\@world xmax 0.002+$maxqpts\n";
    print   "\@world ymin 0\n";
#    print   "\@world ymax 20\n";
    print   "\@default linewidth 2.0\n";
    print   "\@default char size 1.5\n";
    print   "\@autoscale onread yaxes\n";
    
    print   "\@yaxis  bar linewidth 2.0\n";
    print   "\@yaxis label \"$unit_label\"\n";
    print   "\@yaxis label char size 1.5\n";
    print   "\@yaxis ticklabel char size 1.5\n";

#    print   "\@xaxis label \"q\"\n";
    print   "\@xaxis  bar linewidth 2.0\n";
    print   "\@xaxis label char size 1.5\n";
    print   "\@xaxis tick major linewidth 1.6\n";
    print   "\@xaxis tick major grid on\n";
    print   "\@xaxis tick spec type both\n";

    print   "\@xaxis tick spec $#{$qlabels}+1\n";
    if ( $symmetry ne "" ) {
      print   "\@xaxis ticklabel char size 1.5\n";
    } else {
      print   "\@xaxis ticklabel char size 1.0\n";
      print   "\@xaxis ticklabel angle 315\n";
    }
    $i=0;
    for $qpt (@$qlabels) {
      printf STDOUT  "\@xaxis tick major %d,%8.3f\n",$i,$$qpt[0];
      printf STDOUT  "\@xaxis ticklabel %d,\"%s\"\n", $i, qlabel_string($qpt);
      $i++;
    }
    $setnum = 0;
    $filenum  = 0;
    #
    # Write datasets and specific style information
    #
    foreach $i (1..$nbranches ) {
      if ($mono) {
	if ($nspins > 1 ) {
	  if( $i > $nbranches/2) {
	    $colour = 4;
	  } else {
	    $colour = 2;
	  }
	} else {
	  $colour = 1;
	}
	printf STDOUT "\@ G0.S%d line color %d\n",$setnum, $colour;
      }
#      if ($nspins > 1 && $i > $nbranches/2) {
#	printf STDOUT "\@ G0.S%d line linestyle 2\n",$setnum;
#      }
      printf STDOUT "\@target G0.S%d\n",$setnum;
      print "\@type xy\n";
      #
      # Write dataset
      #
      foreach $qpt (0..$nqpts) {
	printf STDOUT "%12.3f %12.3f\n", $$freqs[$qpt][0], $$freqs[$qpt][$i];
      }
      print "&\n";
      $setnum++;
    }

    if( $fermi ne "" ) {
      print "\@ G0.S",$setnum," line linestyle 3\n";
      print "\@ G0.S",$setnum," line color 1\n";
      printf STDOUT "\@target G0.S%d\n",$setnum;
      print "\@type xy\n";
      printf STDOUT "%12.3f %12.3f\n",0,$fermi;
      printf STDOUT "%12.3f %12.3f\n",$maxqpts,$fermi;
      print "&\n";
    }
    #
    # Experimental data is stored as a list of arrays of data [nsets][ndata][2]
    #
    $nsets    = $#$fexptl;

    $symbol = 1;
    for $qpt (0..$nsets) {
      printf STDOUT "\@target G0.S%d\n",$setnum;
      print "\@type xy\n";

      for $datum (0..$#{$fexptl->[$qpt]} ) {
	printf STDOUT "%12.3f %12.3f\n", $$fexptl[$qpt][$datum][0],$$fexptl[$qpt][$datum][1];	
      }
      print "&\n";
      print "\@ G0.S",$setnum," line type 0\n";
      print "\@ G0.S",$setnum," symbol ",$symbol,"\n";
      print "\@ G0.S",$setnum," symbol size 0.5\n";
      $setnum++;
      $symbol++;
    }

    if( $psflag == 1){
      print   "\@hardcopy device \"PS\"\n";
      printf  "\@print to \"%s.ps\"\n\@print\n", $plotfile;
    }
    if( $psflag == 2){
      print   "\@hardcopy device \"EPS\"\n\@device \"EPS\" op \"bbox:tight\"\n";
      printf  "\@print to \"%s.eps\"\n\@print\n", $plotfile;
    }
}

sub gnuplotout {
    my($qlabels, $freqs, $plotfile, $psflag, $unit_label, $title, $fermi, $fexptl ) = @_;
    my($i, $onefile, $filenum, $nbranches, $nqpts, $nqexptl, $setnum, $symbol);
    my($maxqpts, $qpt, $nsets, $datum, $sep);

    $| = 1;
    $nbranches = $#{$freqs->[0]};
    $nqpts     = $#{$freqs};
    $maxqpts   = $$freqs[$nqpts][0];


    if( $psflag == 1){
      print   "set terminal postscript landscape color solid\n";
      printf   "set output \"%s.ps\"\n", $plotfile;
    }
    if( $psflag == 2){
      print   "set terminal postscript eps color solid\n";
      printf   "set output \"%s.eps\"\n", $plotfile;
    }

    print   "set style data lines\n";
    print   "set termoption enhanced\n";
    print   "set termoption font \"Helvetica 16\"\n";
    print   "set title \"",$title,"\"\n";
    print   "set xrange  [0:0.002+$maxqpts]\n";
    print   "set ylabel \"$unit_label\"\n";

    print   "set xtics ";
    $sep = "(";
    for $qpt (@$qlabels) {
	printf STDOUT  "%s \"%s\" %g", $sep, qlabel_string($qpt),$$qpt[0];
	$sep = ",";
    }
    print   ")\n";
    if ( $symmetry ne "" ) {
	print "set xtics \n";
    } else {
	print "set xtics rotate by 315\n";
    }
    print "set grid xtics lt -1\n";

    $setnum = 0;
    $filenum  = 0;
    #
    # Construct plot command
    #
    print   "plot ";
    foreach $i (1..$nbranches ) {
      print ", \\\n" if ($i > 1);
      print   " '-' notitle " ;
      if ($mono) {
	if ($nspins > 1 ) {
	  if( $i > $nbranches/2) {
	    print "lc rgb \"blue\" ";
	  } else {
	    print "lc rgb \"red\" ";
	  }
	} else {
	  print "lc rgb \"black\" ";
	}
      }
    }
    if( $fermi ne "" ) {
      print ", \\\n '-'  title '{/Symbol e}_{/*0.5 F}' lt 0 lc 'black'";
    }
    print   "\n";
    #
    # Write datasets
    #
    foreach $i (1..$nbranches ) {
      foreach $qpt (0..$nqpts) {
	printf  "%12.3f %12.3f\n", $$freqs[$qpt][0], $$freqs[$qpt][$i];
      }
      print "end\n";
    }
    #
    # Data for fermi level indicator
    #
    if( $fermi ne "" ) {
      printf  "%8d %12.3f\n",0,$fermi;
      printf  "%8d %12.3f\n",$maxqpts,$fermi;
      print "end\n";
    }
    #
    # Experimental data is stored as a list of arrays of data [nsets][ndata][2]
    #
#    $nsets    = $#$fexptl;
#
#    $symbol = 1;
#    for $qpt (0..$nsets) {
#      printf STDOUT "\@target G0.S%d\n",$setnum;
#      print "\@type xy\n";
#
#      for $datum (0..$#{$fexptl->[$qpt]} ) {
#	printf STDOUT "%12.3f %12.3f\n", $$fexptl[$qpt][$datum][0],$$fexptl[$qpt][$datum][1];	
#      }
#      print "&\n";
#      print "\@ G0.S",$setnum," line type 0\n";
#      print "\@ G0.S",$setnum," symbol ",$symbol,"\n";
#      print "\@ G0.S",$setnum," symbol size 0.5\n";
#      $setnum++;
#      $symbol++;
#    }
      print   "pause -1\n ";

}

sub write_data {
  my($plotfd, $freqs, $qpts, $qflg) = @_;
  my($n,$q,$freqk);
  $n = 0;
  for $q (0 .. $#$freqs) {
    if( $qflg ) {
      printf $plotfd "%8d    %8.4f %8.4f %8.4f      ",$q,$$qpts[$q][0],$$qpts[$q][1],$$qpts[$q][2];
    } else {
      printf $plotfd "%5d",$n++;
    }

    for $freqk (@{$$freqs[$q]}) {
      printf $plotfd "  %8.3f", $freqk;
    }
    printf $plotfd "\n";
  }
}

#sub dot_product {
#  my ($a, $b) = @_;
#  
#  ${$a}[0]*${$b}[0]+${$a}[1]*${$b}[1]+${$a}[2]*${$b}[2];
#}

sub dot_product {
  my ($a, $b) = @_;
  my ($i,$n,$sum);
  
  $n = $#{$a};
  for $i (0..$n) {
    $sum +=  ${$a}[$i]*${$b}[$i];
  }
  $sum;
}

sub dot_eigens {
  #
  # Compute dot product between eigenvectors.
  # Use real variables not Math::Complex, which is *horribly* slow.
  #
  my ($coords, $a, $b) = @_;
  my ($nions,$ion,$i,$ii,$sum,$pr, $pi);
  my($dot,$avpr,$bvpr,$avpi,$bvpi);

  $nions = $#{$coords};
  $pr = $pi = 0.0;
  for $ion (0..$nions) {
    for $i (0..2) {

      $avpr= $$a[6*$ion+2*$i]; $avpi=$$a[6*$ion+2*$i+1];
      $bvpr= $$b[6*$ion+2*$i]; $bvpi=$$b[6*$ion+2*$i+1];
      # Calculate term A_ni B^*_ni
      $pr += $avpr*$bvpr+$avpi*$bvpi;
      $pi += -$avpr*$bvpi+$avpi*$bvpr;

#      print STDERR $pi,"\n" if fabs($pi) > 0.001;

    }
  }
  sqrt($pr*$pr+$pi*$pi);
}

sub map_exptl {
  #
  # Given a list of the q-vectors at the inflection points of the dispersion curve
  # and a set of experimental data, compute the x-co-ordinate of that datum and
  # rebuild the experimental data list.
  #
  my($exptq, $qlabels) = @_;

  my ($frac, $qdiv, $qcount, $modq, $qsetexp, $qpexpt, $modqmqbeg, $xcoord,$nend,$dot,$collinear,$nbeg);
  my (@qbeg, @qend, @qprev,@deltaq, @qmqbeg, @ndata, @nset);

  # Outer loop is over data sets so we can preserve sets.
  for $qsetexp (@{$exptq} ) {
    @nset=();
    # Loop over direction changes.
    $qcount = 0;
    for $qdiv (@{$qlabels}) {
      @qend = @{$qdiv};
      $nend = shift @qend;
      if( $qcount > 0) {
	@deltaq=($qend[0]-$qbeg[0], $qend[1]-$qbeg[1], $qend[2]-$qbeg[2]);
	$modq=sqrt(dot_product(\@deltaq,\@deltaq));
#	print STDERR "Delta q=(",$deltaq[0],",",$deltaq[1],",",$deltaq[2],"); |dq|=",$modq,"\n";
	if( $modq > $qtol ) {
	  #
	  # Loop over data within set testing against this segment of the graph
	  #
	  for $qpexpt (@{$qsetexp}) {
	    @qmqbeg=(${$qpexpt}[0]-$qbeg[0], ${$qpexpt}[1]-$qbeg[1], ${$qpexpt}[2]-$qbeg[2]);
#	    print STDERR "Testing (",${$qpexpt}[0],",",${$qpexpt}[1],",",${$qpexpt}[2],") against (",$qbeg[0],",",$qbeg[1],",",$qbeg[2],") -> (",$qend[0],",",$qend[1],",",$qend[2],")\n";
	    $modqmqbeg = sqrt(dot_product(\@qmqbeg,\@qmqbeg));
	    $dot=dot_product(\@deltaq,\@qmqbeg);
	    $collinear = (fabs($dot - $modq*$modqmqbeg) < $qtol);
	    $frac = $modqmqbeg/$modq;  # The co-ordinate along qbeg->qend if collinear
	    if( $collinear && $dot >= 0.0 && $frac <= 1.0) {
	      $xcoord = $nbeg + $frac*($nend-$nbeg);
#	      print STDERR "Found q-point (",${$qpexpt}[0],",",${$qpexpt}[1],",",${$qpexpt}[2],") at ",$frac,
#		"(",$xcoord,") along (",
#		  $qbeg[0],",",$qbeg[1],",",$qbeg[2],") -> (",$qend[0],",",$qend[1],",",$qend[2],")\n";
	      push @nset,[$xcoord,${$qpexpt}[3]];
	    }
	  }
	}
      }
      @qbeg = @qend;
      $nbeg = $nend;
      $qcount++;
    }
    push @ndata,[@nset];
  }
  [@ndata];
}

BEGIN { # Static variable declarations
  my(@deltaqprev) = (0.0,0.0,0.0);
  my(@qprev) = (0.0,0.0,0.0);
  my($modqprev) = 0;

sub direction_changed {
  my($qdir,@qpt) = @_;
  my(@deltaq);
  my ($noncollinear,$dot,$modq);

  @deltaq=($qpt[0]-$qprev[0], $qpt[1]-$qprev[1], $qpt[2]-$qprev[2]);
  $dot=$deltaq[0]*$deltaqprev[0]+$deltaq[1]*$deltaqprev[1]+$deltaq[2]*$deltaqprev[2];
  $modq=sqrt($deltaq[0]*$deltaq[0]+$deltaq[1]*$deltaq[1]+$deltaq[2]*$deltaq[2]);
  $noncollinear = (fabs(fabs($dot) - $modq*$modqprev) > $qtol);

  @{$qdir} = @qprev;
  return 0 if ($modq < $qtol); # Repeated q-point. Leave previous direction intact

  @qprev=@qpt;
  @deltaqprev=@deltaq;
  $modqprev = $modq;
  $noncollinear;
}}

sub make_qlabels {
  my($qpts, $abscissa) = @_;
  my (@qlabels,  @qturn);
  my ($nq,$q, $junk);

  $nq=0;
  push @qlabels,[$$abscissa[$nq],@{$qpts[$nq]}];
  for $q (@$qpts) {
    if( direction_changed(\@qturn, @{$q})) {
      push @qlabels,[$$abscissa[$nq-1],@qturn];
    }
    $nq++;
  }
  push @qlabels,[$$abscissa[$nq-1],@{$qpts[$nq-1]}];
  [@qlabels];
}

sub read_exptl {
  my ($file) = @_;
  my (@expdata, @thisline, @qpoint, @dataset);
  my ($f, $nsets);

  open EXPT, "<$file" or die "Failed to open experimental data file $file";

  $nsets = 0;
  $expdata[0] = [0,0,0,0];
  while (<EXPT>) {
    @thisline = split;
    if( $#thisline < 0 ) {
      $nsets++;
      @dataset = ();
    } elsif( $#thisline < 3 ) {
      die "Need at least 4 items on line (qx,qy,qz,f)\n@thisline";
    }
    @qpoint = splice @thisline,0, 3;
    while ( ($f = shift @thisline) ne "") {
      unshift @dataset, [@qpoint, $f] if ($f =~ /$number/);
    }
    $expdata[$nsets] = [@dataset];
  }
  close EXPT;
  [@expdata];
}

sub make_abscissa {
  my ($qpts, $recip) = @_;
  my ($q, $nq, $modq, $qtot);
  my (@deltaq, @delta, @qprev, @abscissa);

  $nq = 0;
  $qtot = 0.0;
  @qprev=(31415927.0,0.0,-271828.0);
  for $q (@$qpts) {
    @delta = ($$q[0]-$qprev[0], $$q[1]-$qprev[1], $$q[2]-$qprev[2]);

    $modq = 0.0;

    #
    # Test for recip lattice translations and skip increment
    #
    my @delta_rem = map {fabs($_ -nearest_int($_))} @delta;
    my $kv_tol = 0.001;

    if( ! (fabs($delta[0])+fabs($delta[1])+fabs($delta[2]) > $kv_tol &&
	$delta_rem[0]+$delta_rem[1]+$delta_rem[2] < $kv_tol ) ){

	@deltaq=&mul3x1t($recip,\@delta);
    
	if( $nq > 0) {
	    $modq=sqrt($deltaq[0]*$deltaq[0]+$deltaq[1]*$deltaq[1]+$deltaq[2]*$deltaq[2]);
	} 
    }
    $qtot += $modq;
    push @abscissa, $qtot;

    $nq++;

    @qprev = @$q;
  }
  [@abscissa];
}

sub MATinv3x3 {
    my ($mat, $inv) = @_;
    my ($bel, $vol, $i, $j);

    @$inv =  ([$$mat[1][1]*$$mat[2][2]-$$mat[2][1]*$$mat[1][2],
	       $$mat[1][2]*$$mat[2][0]-$$mat[2][2]*$$mat[1][0],
	       $$mat[1][0]*$$mat[2][1]-$$mat[2][0]*$$mat[1][1]],
	      [$$mat[2][1]*$$mat[0][2]-$$mat[0][1]*$$mat[2][2],
	       $$mat[2][2]*$$mat[0][0]-$$mat[0][2]*$$mat[2][0],
	       $$mat[2][0]*$$mat[0][1]-$$mat[0][0]*$$mat[2][1]],
	      [$$mat[0][1]*$$mat[1][2]-$$mat[1][1]*$$mat[0][2],
	       $$mat[0][2]*$$mat[1][0]-$$mat[1][2]*$$mat[0][0],
	       $$mat[0][0]*$$mat[1][1]-$$mat[1][0]*$$mat[0][1]]);

    $vol=$$mat[0][0]*$$inv[0][0]+$$mat[0][1]*$$inv[0][1]+$$mat[0][2]*$$inv[0][2];

    die "Error - cell volume <= 0.   Problem with cell parameters? " if (fabs($vol) <= 0.0 );

    for $i (0..2) {
      for $j (0..2) {
	$inv->[$i][$j] = $inv->[$i][$j] * (2.0*$pi/$vol);
      }
    }
}

sub mul3x1 {
  my($mat, $vec)= @_;
  my ($i,$j,$n,$sum,@res);

  $n = $#{$vec};

  for $i (0..$n) {
    for $j (0..$n) {
      $res[$i] += $mat->[$i][$j]*$vec->[$j];
    }
  }
  @res;
}

sub mul3x1t {
  my($mat, $vec)= @_;
  my ($i,$j,$n,$sum,@res);

  $n = $#{$vec};

  for $i (0..$n) {
    for $j (0..$n) {
      $res[$i] += $mat->[$j][$i]*$vec->[$j];
    }
  }
  @res;
}

sub MATtoABC {
  my($mat) = @_;
  my ($a, $b, $c, $aa, $ba, $ca, $ag, $bg, $cg);
  my $RTOD = 180.0 / $pi;

  $a = sqrt($$mat[0][0] ** 2 + $$mat[0][1] ** 2 + $$mat[0][2] ** 2);
  $b = sqrt($$mat[1][0] ** 2 + $$mat[1][1] ** 2 + $$mat[1][2] ** 2);
  $c = sqrt($$mat[2][0] ** 2 + $$mat[2][1] ** 2 + $$mat[2][2] ** 2);
  
  $ag = $RTOD * &acos(($$mat[1][0] * $$mat[2][0] + $$mat[1][1] * $$mat[2][1] 
		       + $$mat[1][2] * $$mat[2][2]) / ($c * $b));
  $bg = $RTOD * &acos(($$mat[0][0] * $$mat[2][0] + $$mat[0][1] * $$mat[2][1] 
		       + $$mat[0][2] * $$mat[2][2]) / ($a * $c));
  $cg = $RTOD * &acos(($$mat[0][0] * $$mat[1][0] + $$mat[0][1] * $$mat[1][1] 
		       + $$mat[0][2] * $$mat[1][2]) / ($a * $b));
  return ($a, $b, $c, $ag, $bg, $cg);
}

sub nearest_int {
    floor($_[0] + 0.5);
}
