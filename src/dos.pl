#! /usr/bin/env perl
#
# Parse a "<>.castep" or "<>.phonon" or "<>.bands" output file from
# New CASTEP for vibrational frequency data and output an xmgrace plot
# of the electronic or vibrational DOS.
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
   printf STDERR "Usage: dos.pl [-xg] [-ps|-eps] [-np] <seed>.castep|<seed>.phonon ... \n";
   printf STDERR "       Extract phonon or bandstructure data from .castep, .phonon or .bands files";
   printf STDERR "       and optionally prepare a DOS plot using XMGRACE as a backend.";
   printf STDERR "    -xg        Write a script and invoke GRACE to plot data.\n";
   printf STDERR "    -gp        Write a script and invoke GNUPLOT to plot data.\n";
   printf STDERR "    -ps        Invoke GRACE to plot data and write as a PostScript file.\n";
   printf STDERR "    -eps       Invoke GRACE to plot data and write as an encapsulated.\n               PostScript (EPS) file.\n";
   printf STDERR "    -np        Do not plot data, write a GRACE script.\n";
   printf STDERR "    -bs        Read band-structure from <>.castep or <>.bands.\n";
   printf STDERR "    -mirror    Plot spin-polarized electronic DOS using \"mirror\" plot.\n";
   printf STDERR "    -b w       Set histogram resolution for binning (eV or cm**-1).\n";
   printf STDERR "    -ir        Extract ir intensities and model (fundamentals-only) ir spectrum from .phonon.\n";
   printf STDERR "    -raman     Extract raman intensities and model (fundamentals-only) raman spectrum from .phonon.\n";
   printf STDERR "    -temp T    Temperature to use (in raman spectrum modelling).\n";
   printf STDERR "    -expt FILE Read experimental data from EXPT and overplot.\n";
   printf STDERR "    -dat       Reread standard output from previous run and plot.\n";
   printf STDERR "    -w s       Set Gaussian/Lorentzian FWHM for broadening.\n";
   printf STDERR "    -lorentz   Use Lorentzian broadening instead of Gaussian\n";
   printf STDERR "    -units s      Convert output to specified units for plotting.\n";
   printf STDERR "    -v         Be verbose about progress\n";
   printf STDERR "    -z         Print zero-point energy\n";
die;
}

my $number  = qr/-?(?:\d+\.?\d*|\d*\.?\d+)(?:[Ee][+-]?\d{1-3})?/o;
my $fnumber = qr/-?(?:\d+\.?\d*|\d*\.?\d+)/o;

my $qtol = 5.0e-6;
my $fzerotol = 3.0;
my $bwidth = 1.0;
my $gwidth = 10.0;
my $pi=3.14159265358979;

my $castep_phonon_re = qr/^ +\+ +q-pt= +\d+ \( *( *$fnumber) *($fnumber) *($fnumber)\) +($fnumber) +\+/o;
my $dot_phonon_re    = '';

my $castep_bs_re = qr/^  \+ +Spin=(\d) kpt= +(\d+) \( *( *$fnumber) *($fnumber) *($fnumber)\) +kpt-group= +\d +\+/o;
my $dot_bands_re    = '';

my $castep_re;

my ($grflag, $gpflag, $plotflag, $datflag,  $irflag, $ramanflag, $psflag, $qlabels, $exptfile, $exptq, $abscissa,$verbose) = ("", "", "","","","","","","");
my ($title, $fileroot, $units, $xlabel, $ylabel, $fermi_u, $fermi_d, $i,$freq,$exptn, $wt, $b, $c, $temperature, $lorentzian, $zpe, $nqlast,$spin_polarised);

my (@freqs, @freqs_d, @qpts, @weights, @dos, @dos_u, @dos_d, @ir_intensities, @r_intensities, @headers, $base);

my ($opt_xg, $opt_gp, $opt_np, $opt_ps, $opt_eps, $opt_bs, $opt_mirror, $opt_bw,$opt_ir, $opt_raman, $opt_temp, $opt_lorentz,  $opt_dat, $opt_expt, $opt_gwidth,$opt_units, $opt_help,$opt_v,$opt_z) = (0,0,0,0,0,0,0,"",0,0,0,0);
my ($is_dot_castep, $i);
my ($readdos, $num_dos, $dos_cpt, $unit_label, $unit_conv, $input_unit);

&GetOptions("xg"   => \$opt_xg, 
	    "gp"   => \$opt_gp, 
	    "np"   => \$opt_np, 
	    "ps"   => \$opt_ps, 
	    "eps"  => \$opt_eps, 
	    "bs"   => \$opt_bs, 
	    "mirror"   => \$opt_mirror, 
	    "b=f"  => \$opt_bw,
	    "ir"   => \$opt_ir, 
	    "raman" => \$opt_raman, 
	    "temp=f" => \$opt_temp,
	    "dat"  => \$opt_dat, 
	    "expt=s" => \$opt_expt,
	    "lorentz" => \$opt_lorentz, 
	    "w=s"  => \$opt_gwidth, 
	    "units=s"   => \$opt_units,
	    "h|help"    => \$opt_help,
	    "v"    => \$opt_v,
	    "z"    => \$opt_z) || &usage();

if ($opt_help || $opt_ps && $opt_eps) {&usage()};

$grflag= $opt_xg if $opt_xg;
$gpflag= $opt_gp if $opt_gp;
$plotflag = 1;
$plotflag= 0 if $opt_np;
$psflag = 0;
$psflag= 1 if $opt_ps;
$psflag= 2 if $opt_eps;
$grflag = 1 if $psflag && ! ($grflag || $gpflag) ;
$exptfile = "";
$exptfile = $opt_expt if $opt_expt;
$datflag = 1 if $opt_dat;
$gwidth = $opt_gwidth if $opt_gwidth >= 0;
$verbose = 0;
$verbose = 1 if $opt_v;
$opt_bs = 1 if( $#ARGV >= 0 && $ARGV[0]=~ /\.bands$/ );
$opt_bs = 1 if $opt_mirror;
$opt_bs = 0 if( $#ARGV >= 0 && $ARGV[0]=~ /\.phonon$/ );
$irflag = $opt_ir;
$ramanflag = $opt_raman;
$temperature = 300;
$temperature=$opt_temp if $opt_temp;
$lorentzian = $opt_lorentz;

$readdos = 0;
$readdos = 1 if $ARGV[0] =~/phonon_dos/;

if ($#ARGV >= 0) {
  $title=$ARGV[0];
  $title=~s/\.(phonon|castep|bands|phonon_dos)//;
} else {
  $title = "";
}
$fileroot=$title;

#
# Set up units for output from options and defaults
#
my %unit_scale = (
    # Energies and frequencies - relative to default eV
    "mev"  => 1000.0,
    "thz"  => 241.7989348,
    "cm-1" => 8065.73,
    "ev"   => 1.0,
    "Ha"   => 1/27.21138505,
    "mha"  => 1/0.02721138505,
    "ry"   => 1/13.605692525,
    "mry"  => 1/0.013605692525
    );

$units = $opt_units;

if( $opt_bs ) {
  $units = "eV" if( $opt_units eq "" );
  $input_unit = 1.0;
  $castep_re = $castep_bs_re;
  if( $grflag ) {
    $ylabel = 'g(\\f{Symbol}e\\f{})';
  } elsif ($gpflag ) {
    $ylabel = 'g({/Symbol e})';
  }
  $gwidth = 0.1;
  $gwidth = $opt_gwidth if $opt_gwidth;
  $bwidth = 0.05;
  $fzerotol = 0.0;
} else {
  $units = "cm-1" if( $opt_units eq "" );
  $input_unit = 8065.73;
  $castep_re = $castep_phonon_re;
  if( $grflag ) {
    $ylabel = 'g(\\f{Symbol}w\\f{})';
  } elsif ($gpflag ) {
    $ylabel = 'g({/Symbol w})';
  }
}

$unit_conv = $unit_scale{lc($units)} / $input_unit;

#
# Back-scale supplied gwidth to default units
#
$gwidth /= $unit_conv if( $opt_gwidth );

$unit_label = "default";
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
$xlabel = $unit_label;

$bwidth = $opt_bw if( $opt_bw );

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

if ( $readdos ) {

    read_dot_phonon_dos(\@dos, \$base, \$bwidth, \@headers);

} else {
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
	reread_data(\@freqs,\@qpts);
    } elsif ( $is_dot_castep ) {
	if(  $opt_bs ) {
	    read_dot_castep_bands(\@freqs,\@freqs_d,\@qpts, \$fermi_u, \$fermi_d, \@weights);
	} else {
	    read_dot_castep_phonon(\@freqs,\@qpts, \@weights, \@ir_intensities, \@r_intensities);
	}
    } else {
	if(  $opt_bs ) {
	    read_dot_bands(\@freqs,\@freqs_d,\@qpts, \$fermi_u, \$fermi_u, \@weights);
	} else {
	    read_dot_phonon(\@freqs,\@qpts, \@weights, \@ir_intensities, \@r_intensities);
	}
    }

    $spin_polarised = $#{$freqs_d[0]} > 0;

    $base = 1.0e21;
    if( $irflag ) {
	compute_dos(\@freqs, \@weights, \@dos_u, $bwidth, $gwidth, \$base, \@ir_intensities);
    } elsif( $ramanflag ) {
	compute_raman(\@freqs, \@weights, \@dos_u, $bwidth, $gwidth, \$base, \@r_intensities);
    } else {
	$zpe = compute_zpe(\@freqs, \@weights);
	printf STDERR "Zero Point Energy = %12.6f eV\n", $zpe if $opt_z;
	compute_dos(\@freqs, \@weights, \@dos_u, $bwidth, $gwidth, \$base);
	compute_dos(\@freqs_d, \@weights, \@dos_d, $bwidth, $gwidth, \$base) if ($opt_bs && $spin_polarised) ;
    }
    
    @dos = (\@dos_u, \@dos_d);
}

# Convert to units specified
for $dos_cpt (@dos) {
  map {$_ /= $unit_conv}  @{$dos_cpt};
}
$bwidth *= $unit_conv;
$base *= $unit_conv;

if( $grflag ) {
  if ($spin_polarised) {@headers = ("alpha","beta");}
  if(  $opt_bs ) {
    graceout(\@dos, $base, $bwidth, $fileroot, $psflag, $xlabel, $ylabel, $title, $fermi_u, "", \@headers);
  } else {
    graceout(\@dos, $base, $bwidth, $fileroot, $psflag, $xlabel, $ylabel, $title,"" ,$exptn, \@headers);
  }
} elsif( $gpflag ) {
  if(  $opt_bs ) {
    gnuplotout(\@dos, $base, $bwidth, $fileroot, $psflag, $xlabel, $ylabel, $title, $fermi_u);
  } else {
    gnuplotout(\@dos, $base, $bwidth, $fileroot, $psflag, $xlabel, $ylabel, $title,"" ,$exptn);
  }

} else {
  for $b (0..$#{$dos[0]}) {
      printf "%12.6f ", $b*$bwidth+$base;
      for $c (0..$#dos) {printf "  %12.6f", $dos[$c][$b]; }
      print "\n";
  }
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
  my($freqs, $qpts, $weights, $intens, $rintens) = @_;
  my(@freqk,@q,@intensk,@rintensk);
  my ($weight);

  while ( <>) {
    #
    # <>.castep
    #
    if (/$castep_phonon_re/o) {
      @q=($1,$2,$3);
      $weight = $4;
      print STDERR "Reading $q[0] $q[1] $q[2] $weight\n" if $verbose;
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
      @intensk=();
      @rintensk=();

      while (<>) {
	if (/^ +\+ +\d+ +($fnumber)( +\w)? *($fnumber)?( +[YN])? *($fnumber)?( +[YN])? *\+/) { 
	  push @freqk, $1;
	  push @intensk, $3;
	  push @rintensk, $5;
#	} elsif (/^  \+ +q->0 along \( *( *$fnumber) *($fnumber) *($fnumber)\) +\+/ ) {
	} elsif (/^ +\+ -+ \+/) {
	  last;
	} elsif (/^ +\+ +.*\+/) {
	} else {
	  last;
	}
      }
      push @$freqs, [@freqk];
      push @$intens, [@intensk];
      push @$rintens, [@rintensk];
      push @$qpts, [@q];
      push @$weights, $weight;
      last if eof ARGV;
    }
  }
}

sub read_dot_castep_bands {
  my($freqs_u, $freqs_d, $qpts, $fermi_u, $fermi_d, $weights) = @_;
  my(@freqk,@q,@kp_map,$kp_count,$spin,$kpt,$weight,$nk);
  
  $kp_count = 0;
  while ( <>) {
    #
    # <>.castep
    #
    $$fermi_u = $1 if /^  \+  Fermi energy for spin  up  electrons is:\s+ ($number)[^+]+\+/;
    $$fermi_d = $1 if /^  \+  Fermi energy for spin down electrons is:\s+ ($number)[^+]+\+/;
    if (/$castep_bs_re/o) {
      ($spin,$kpt,@q,$weight)=($1,$2,$3,$4,$5,$6);
      print STDERR "Reading Spin=$spin;kpt=$kpt  ($q[0],$q[1],$q[2]) $weight\n" if $verbose;
      #print "   ";
      last if eof ARGV;
      $_ = <>;
      last if eof ARGV;
      $_ = <> if /^  \+ -+ \+/;
      @freqk=();

      while (<>) {
	if (/^  \+ +\d+ +($fnumber)( +\w)?( +$fnumber)* *\+/) { # This matches exptl IR -active format
	  push @freqk, $1;
#	} elsif (/^  \+ +q->0 along \( *( *$fnumber) *($fnumber) *($fnumber)\) +\+/ ) {
	} elsif (/^  \+ -+ \+/) {
	  last;
	} elsif (/^  \+ +.*\+/) {
	} else {
	  last;
	}
      }
      print STDERR @freqk,"\n" if $verbose;
      #
      # Keep track of kpt ordering for spin=2.  Assumes all spin=1 precede spin=2.
      #
      if( $spin == 1 ) {
	$kp_map[$kpt] = $kp_count++;
	push @$freqs_u, [@freqk];
	push @$qpts, [@q];
	push @$weights, 1.0;
      } else {
	push @{$freqs_d[$kp_map[$kpt]]}, @freqk;
      }
      last if eof ARGV;
    }
  }
  for $nk (0..$#weights) {
    $weights[$nk] = 1.0/$kp_count;
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
  my($freqs, $qpts, $weights, $intens, $rintens) = @_;
  my(@freqk,@freqk_r,@freqk_o,@q,@eigvec, @oeigvec, @qmapt, @qmap, @intensk, @rintensk);
  my(@coords,@xyz,@corrmat);
  my($x,$y,$z,$nions,$nmodes, $ion,$aty, $mass, $count, $mode, $omode,$dot);
  my($qmode,$qomode,$dot_same,$savedot,$maxdot,$save,$qmaxdot,$weight);

  #
  # Read frequencies from <>.phonon file
  #
  # Also reads eigenvectors and attempts to join up branches
  # on the dispersion curve using a heuristic algorithm based
  # assuming close to orthonormality of eigenvectors at
  # adjacent q-points.
  #
  @$weights = ();
  @$qpts    = ();
  @$freqs   = ();
  @$intens  = ();
  while (<>) {
    #
    # <>.phonon
    #
    $nions = $1 if /^ Number of ions\s*(\d+)/;
    $nmodes = $1 if /^ Number of branches\s*(\d+)/;
    if( / Fractional Co-ordinates/../ end.*header/i) {
      if( /^\s+\d+(\s+$fnumber){3}\s+[A-Za-z]{1,2}\s+$fnumber/ ) {
	($ion,$x,$y,$z,$aty,$mass) = split;
	push @coords, [$x,$y,$z];
      }
    }
    #
    # Found start of block of frequencies
    #
    if (/^ +q-pt= +(\d+) +($fnumber) +($fnumber) +($fnumber) *($fnumber){0,4}/) {
#    if (/^ +q-pt= +\d+ +($fnumber) *($fnumber) *($fnumber) ( *$fnumber){0,4}$/) {
      @q=($2,$3,$4);
      $weight = $5;
      #
      # Test for repeated q-point and assign zero weight to copies
      #
      $weight = 0.0 if( $1 == $nqlast );
      $nqlast = $1;
      print STDERR " ",$q[0]," ",$q[1]," ",$q[2]," ",$weight,"\n" if $verbose;
      #print "   ";
      @intensk=();
      @rintensk=();
      @freqk_r=();

      while (<>) {
	if (/^ +\d+ +($fnumber) +($fnumber) +($fnumber)/) {
	  push @freqk_r, $1;
	  push @intensk, $2;
	  push @rintensk, $3;
	} elsif (/^ +\d+ +($fnumber) +($fnumber)/) {
	  push @freqk_r, $1;
	  push @intensk, $2;
	} elsif (/^ +\d+ +($fnumber) */) {
	  push @freqk_r, $1;
	} else {
	  last;
	}
      }
#      print STDERR @freqk_r,"\n";
      push @$qpts, [@q];
      push @$freqs, [@freqk_r];
      push @$intens, [@intensk];
      push @$rintens, [@rintensk];
      push @$weights, $weight;
    }
  }
}

sub read_dot_phonon_dos {
  my($dos, $base, $bwidth, $headers) = @_;

  my(@species_dos);
  my(@coords,@xyz,@corrmat);
  my($x,$y,$z,$nions, $nmodes, $nspecies, $ion,$aty, $mass, $count, $freq, $last_freq, $total, $dos_cpt);

  #
  # Read DOS frequencies from <>.phonon_dos file
  #

  $total = 0;
  $$base = 1e21;

  @$dos=();

  while (<>) {
    #
    # <>.phonon_dos
    #
    $nions = $1 if /^ Number of ions\s*(\d+)/;
    $nmodes = $1 if /^ Number of branches\s*(\d+)/;
    $nspecies = $1 if /^ Number of species\s*(\d+)/;
    
    if( / Fractional Co-ordinates/../ end.*header/i) {
      if( /^\s+\d+(\s+$fnumber){3}\s+[A-Za-z]{1,2}\s+$fnumber/ ) {
	($ion,$x,$y,$z,$aty,$mass) = split;
	push @coords, [$x,$y,$z];
      }
    }
    #
    # Found start of block of frequencies
    #
    if ( /^ *BEGIN DOS/ ) {
	@$headers = split;
	for $i ( 1..5 ) { shift @$headers; }
	unshift @$headers, "Total";
	for $i ( 0..$#{$headers} ) { push @$dos, [];}
    }

    if ( /^ *BEGIN DOS/ ..  /^ *END DOS/) {
	if (/^ *($fnumber){2,}/) {
	    @species_dos = split;
	    $last_freq = $freq;
	    $freq = shift @species_dos;
	    for $dos_cpt (0 .. $#species_dos) {push @{$dos[$dos_cpt]}, $species_dos[$dos_cpt];}
	    $total += $species_dos[0];
	    $$base = $freq if ($freq < $$base) ;
	    $$bwidth = $freq - $last_freq; # Find bin width
	}
    }
  }
#  printf STDERR "Found DOS for ";
#  for $header (@headers) {print STDERR $header,", "; };
#  printf STDERR "\n";
#  printf STDERR "WIDTH=$$bwidth  BASE=$$base\n";
  printf STDERR "Integrated DOS = %12.4f\n",$total*$$bwidth;
}

sub read_dot_bands {
  my($freqs_u, $freqs_d, $qpts, $fermi, $fermi1, $weights) = @_;
  my(@eigk,@k, @freq_list, @qpt_list, @nk_list, @wt_list);
  my($wt, $first, $Hartree,$nk,$ink,$ink_save, $ncount, $cpt, $spin_polarised);

  $Hartree =27.2114;
  $first = 1;
  $nk = -1;
  $ncount = 0;
  $spin_polarised = 0;
  $cpt = 0;
  #
  # Pass 1 - read the data
  while (<>) {
    #
    # <>.bands
    #
    chop;
    if (/^K-point +(\d+) +($fnumber) *($fnumber) *($fnumber) *($fnumber)/) {
      if( ! $first ) {
	push @{$freq_list[$cpt]}, [@eigk];
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
      $cpt = 0;
    } elsif (/^Spin component 1/) {
    } elsif (/^Spin component 2/) {
      push @{$freq_list[0]}, [@eigk];
      print STDERR "Read K-point $nk, ($qpt_list[$ncount-1][0],$qpt_list[$ncount-1][1],$qpt_list[$ncount-1][2]), spin 2\n" if $verbose;
      @eigk=();

      $cpt = 1;
      $spin_polarised = 1;
    } elsif (/^ +($fnumber)\s*$/) {
      push @eigk, $1*$Hartree;
    } elsif ( /Fermi energy \(in atomic units\) +($fnumber)/ ) {
      $$fermi = $1 * $Hartree;
    } elsif ( /Fermi energies \(in atomic units\) +($fnumber) +($fnumber)/ ) {
      $$fermi = $1 * $Hartree;
      $$fermi1 = $1 * $Hartree;
    }
  }
  push @{$freq_list[$cpt]}, [@eigk];
  if( $cpt - 0 > 0 ) { # Force conversion to F.P.
    #
    # Already have k-point and weight if this is second spin component
    #
    push @qpt_list,  [@k];
    push @nk_list,   $nk;
    push @wt_list,   $wt;
  }

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
      $$freqs_u[$nk] = $freq_list[0][$ink];
      $$freqs_d[$nk] = $freq_list[1][$ink] if $spin_polarised;
      
      $$qpts[$nk] =  $qpt_list[$ink];
      $$weights[$nk] = $wt_list[$ink];
    }
  }
  print STDERR "Spin_polarised=",$spin_polarised,"\n";
  #
  # Due to a CASTEP bug some state may be double-labelled. In that case we have a "spare" set
  # of frequencies in $ink_save.  See if we can guess where to put it.
  for $ink (1..$#$qpts) {
    if( $#{$$freqs_u[$ink]} < 1 ) {  # Aha!
    print STDERR "Placing missing K-point $ink, ($qpt_list[$ink_save][0],$qpt_list[$ink_save][1],$qpt_list[$ink_save][2])\n" if $verbose;
      
      $$freqs_u[$ink] = $freq_list[0][$ink_save];
      $$freqs_d[$ink] = $freq_list[1][$ink_save] if $spin_polarised;
      $$qpts[$ink] =  $qpt_list[$ink_save];
    }
  }

}

sub ftrim {
  my($val,$prec) = @_;
  my($str,$w);
  
  $w=$prec+2;
  $str = sprintf "%${w}.${prec}f",$val;
  $str =~ s/0+$//;
  $str;
}

sub graceout {
    my($dos, $base, $bwidth, $plotfile, $psflag, $xlabel, $ylabel, $title, $fermi, $fexptl, $headers ) = @_;
    my($i,  $setnum);
    my($xmax, $ymax, $n, $ncpts, $sign);
    my %special = {"alpha" => "\\f{Symbol}a\\f{}", "beta" => "\\f{Symbol}b\\f{}"};

    $| = 1;
    $xmax = $base+$bwidth*$#{$dos->[0]};

    print   "\@timestamp def \"",scalar(localtime),"\"\n";
    print   "\@with g0\n";
    print   "\@title \"",$title,"\"\n";
    print   "\@world xmin $base\n";
    print   "\@world xmax $xmax\n";
    print   "\@view  ymin 0.35\n";
    print   "\@view  xmax 0.75\n";
    print   "\@legend  0.625, 0.825\n";
    print   "\@autoscale onread xyaxes\n";
    
    print   "\@xaxis label \"$xlabel\"\n";
    print   "\@yaxis label \"$ylabel\"\n";

    $setnum=0;
    if( $#{$dos->[1]} > 0 && $opt_bs) {
      print "\@ G0.S",$setnum," line color 2\n";
    }
    for $i (0..$#{$headers} ) {
	$$headers[$i] = $special{$$headers[$i]} if ($special{$$headers[$i]} != "");
    }
    print "\@ G0.S",$setnum," legend \"$$headers[0]\"\n";
    printf STDOUT "\@target G0.S%d\n",$setnum;
    print "\@type xy\n";

    $ymax = 0.0;
    foreach $n (0..$#{$dos->[0]}) {
      printf STDOUT "%15.6f %15.6f\n", $n*$bwidth+$base, ${$dos->[0]}[$n];
      $ymax =  ${$dos->[0]}[$n] if  ${$dos->[0]}[$n] > $ymax;
    }
    print "&\n";
    #
    # More components?
    #
    for $i (1..$#{$dos} ) {
      $setnum++;

      $sign = 1;
      if( $opt_mirror ){
	$sign = -1;

	print "\@altxaxis  on\n";
	print "\@altxaxis  type zero true\n";
	print "\@altxaxis  tick off\n";
	print "\@altxaxis  ticklabel off\n";

      }

      if( $opt_bs ) {
	  print "\@ G0.S",$setnum," line color ",4,"\n";
      } else {
	  print "\@ G0.S",$setnum," line color ",$i+1,"\n";
      }
      print "\@ G0.S",$setnum," legend \"$$headers[$i]\"\n";
      printf STDOUT "\@target G0.S%d\n",$setnum;
      print "\@type xy\n";

      foreach $n (0..$#{$dos->[$i]}) {
	printf STDOUT "%15.6f %15.6f\n", $n*$bwidth+$base, $sign*${$dos->[$i]}[$n];
	$ymax =  ${$dos->[$i]}[$n] if  ${$dos->[$i]}[$n] > $ymax;
      }
      print "&\n";

    }
    $setnum++;

    if( $fermi ne "" ) {
      print "\@ G0.S",$setnum," line linestyle 3\n";
      print "\@ G0.S",$setnum," line color 1\n";
      print "\@ G0.S",$setnum," legend \"\\f{Symbol}e\\f{}\\sF\\N\"\n";
      printf STDOUT "\@target G0.S%d\n",$setnum;
      print "\@type xy\n";
      if( $opt_mirror ) {
	printf STDOUT "%15.6f %15.6f\n",$fermi,-$ymax;
      } else {
	printf STDOUT "%15.6f %15.6f\n",$fermi,0;
      }
      printf STDOUT "%15.6f %15.6f\n",$fermi,$ymax;
      print "&\n";
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
    my($dos, $base, $bwidth, $plotfile, $psflag, $xlabel, $ylabel, $title, $fermi, $fexptl ) = @_;
    my($i,   $setnum);
    my($xmax, $ymax, $n, $sign);

    $| = 1;
    $xmax = $base+$bwidth*$#{$dos->[0]};
    $xmax = $fermi + 1 if ($fermi ne "" && $fermi + 1 > $xmax) ;

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
    print   "set title \"",$title,"\"\n";
    print   "set xlabel \"$xlabel\"\n";
    print   "set ylabel \"$ylabel\"\n";
    print   "set xrange  [$base:$xmax]\n";

    if( $#{$dos->[1]} > 0 ) {
      print   "plot '-' title '{/Symbol a}', '-' title '{/Symbol b}' lt 3" ;

      if( $opt_mirror ){
	print ", '-' title '' lt 1 lc 'black'";
      }
    } else {
      print   "plot '-' title '$ylabel' lt 1 lc 'black'" ;
    }
    if( $fermi ne "" ) {
      print  ", '-' title '{/Symbol e}_{/*0.5 F}' lt 0 lc 'black'" ;
    }
    print "\n";
    #
    # Write datasets
    #
    $ymax = 0;
    foreach $n (0..$#{$dos->[0]}) {
      printf "%15.6f %15.6f\n", $n*$bwidth+$base, ${$dos->[0]}[$n];
      $ymax =  ${$dos->[0]}[$n] if  ${$dos->[0]}[$n] > $ymax;
    }
    print "end\n";
    if( $#{$dos->[1]} > 0 ) {
      $sign = 1;
      $sign = -1.0 if( $opt_mirror );

      foreach $n (0..$#{$dos->[1]}) {
	printf "%15.6f %15.6f\n", $n*$bwidth+$base, $sign*${$dos->[1]}[$n];
	$ymax =  ${$dos->[1]}[$n] if  ${$dos->[1]}[$n] > $ymax;
      }
      print "end\n";
    }
    #
    # Data for fermi level indicator
    #
    if( $fermi ne "" ) {
      if( $opt_mirror ) {
	printf  "%15.6f %15.6f\n",$base, 0;
	printf  "%15.6f %15.6f\n",$xmax, 0;
	print "end\n";

	printf  "%15.6f %15.6f\n",$fermi,-$ymax;
      } else {
	printf  "%15.6f %15.6f\n",$fermi,0;
      }
      printf  "%15.6f %15.6f\n",$fermi,$ymax;
    }
    print "end\n";

    print   "\npause -1\n";

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
      printf $plotfd "  %12.6f", $freqk;
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
    while ( $f = shift @thisline) {
      unshift @dataset, [@qpoint, $f] if ($f =~ /$number/);
    }
    $expdata[$nsets] = [@dataset];
  }
  close EXPT;
  [@expdata];
}

sub compute_zpe {
  my ($freqs, $weights) = @_;
  my ($nq, $mode);
  my $zpe  = 0.0;
  my $cm_to_eV = 1.239842e-4;

  for $nq (0..$#{$freqs}) {
#    printf STDERR "Q-pt %d weight %f\n",$nq+1,$$weights[$nq];
    for $mode (0..$#{$$freqs[$nq]}) {
      $freq = ${$freqs}[$nq]->[$mode];
#      printf STDERR "Mode %d freq %f\n",$mode+1,$freq;
      $zpe += $freq*$$weights[$nq];
    }
  }

  $zpe *= 0.5*$cm_to_eV;
  $zpe;
}

sub compute_dos {
  my ($freqs, $weights, $dos, $bwidth, $gwidth, $baseptr, $intensities) = @_;

  my ($nq, $q, $mode, $freq, $bin, $base, $g,$h, $ngauss, $nlorentz, $intensity,$gammaby2,$sigma);
  my ( $gauss, $lorentz, $h0, $h1);
  my (@hist);

  if ( $$baseptr > 1.0e20 ) {
    $base = $$baseptr;

    for $nq (0..$#{$freqs}) {
      for $mode (0..$#{$$freqs[$nq]}) {
	$freq = ${$freqs}[$nq][$mode];
	$base = $freq if $freq < $base;
      }
    }
    $$baseptr = $base;
  } else {
    $base = $$baseptr;
  }
  for $nq (0..$#{$freqs}) {
    for $mode (0..$#{$$freqs[$nq]}) {
      $freq = ${$freqs}[$nq]->[$mode];
      $bin = ($freq-$base)/$bwidth;
      $intensity = 1.0;
      $intensity = 0.0 if abs($freq) < $fzerotol;
      $intensity =  ${$intensities}[$nq]->[$mode] if ref $intensities ne "";
      $hist[$bin] += $$weights[$nq]*$intensity;
    }
  }

  $ngauss = 3.0*$gwidth/$bwidth;  # 3 sd should do
  $sigma = $gwidth/2.354;
  $nlorentz = 25.0*$gwidth/$bwidth;  # 5 widths?
  $gammaby2 = $gwidth/2;

  if( $gwidth < $bwidth ) { # No broadening
    for $h (0..$#hist) {
      $$dos[$h] = $hist[$h]/$bwidth;
    }
  } else {
    if( $lorentzian ) { 
      for $g (-$nlorentz..$nlorentz) {
	$lorentz = $gammaby2/($g**2+$gammaby2**2)/$pi;
	$h0 = $g;
	$h0 = 0 if $h0 < 0;
	$h1 = $#hist + $g;
	
	for $h ($h0..$h1) {
	  $$dos[$h] += $hist[$h-$g]*$lorentz;
	}
      }
    } else {
      for $g (-$ngauss..$ngauss) {
	$gauss = exp( - ($g*$bwidth)**2/(2*$sigma**2))/ (sqrt(2*$pi)*$sigma);
	$h0 = $g;
	$h0 = 0 if $h0 < 0;
	$h1 = $#hist + $g;
	
	for $h ($h0..$h1) {
	  $$dos[$h] += $hist[$h-$g]*$gauss;
	}
      }
    }
  }

  my $total = 0.0;
  for $h (0..$#$dos) {
     $total += $$dos[$h];
  }
  printf STDERR "Integrated DOS = %12.4f\n",$total*$bwidth;

  if ( $verbose ) {
    for $b (0..$#$dos) {
      printf STDERR "%4d %12.4f\n", $hist[$b], $$dos[$b];
    }
  }
  
}

sub compute_raman {
  my ($freqs, $weights, $dos, $bwidth, $gwidth, $baseptr, $intensities) = @_;

  my $c = 299792458;
  my $laser_wavelength = 514.5e-9; # Ar at 514.5 nm
  my $planck = 6.6260755e-34;
  my $cm1k = 1.438769; # cm(-1) => K conversion
  my ($n,$freq,$mode, $factor, $bose_occ, $cross_section, @cross_secs);

  $factor = pow(2*$pi/$laser_wavelength,4)*$planck/(8*$pi**2*45)*1e12;

#  print STDERR "MULT=$factor\n";

  @cross_secs = ();
  $n=$#{$$freqs[0]};
  for $mode (0..$n) {
    $freq = ${$freqs}[0][$mode];
    if( $freq > $fzerotol ){
      $bose_occ = 1.0/(exp($cm1k*$freq/$temperature)-1);
      $cross_section = $factor/$freq * (1 + $bose_occ)*$$intensities[0][$mode];
    } else {
      $bose_occ = 1.0;
      $cross_section = 0;
    }
    push @cross_secs, $cross_section;
#    print STDERR "f=$freq;  occ=$bose_occ; act=$$intensities[0][$mode]; ds/dO= $cross_section\n";
  }
  compute_dos($freqs, $weights, $dos, $bwidth, $gwidth, $baseptr, [[@cross_secs]]);
  
}
