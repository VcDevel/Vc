/*
    Copyright (C) 2009 Matthias Kretz <kretz@kde.org>

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation; either version 2 of
    the License, or (at your option) version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
    02110-1301, USA.

*/

#include <unistd.h>
#include <sched.h>
#include <signal.h>
#include <cstdio>
#include <cstring>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/select.h>

int main(int, char **argv)
{
  sched_param param;
  sched_getparam( 0, &param );
  param.sched_priority = 2;
  if ( 0 != sched_setscheduler( 0, SCHED_FIFO, &param ) ) {
    perror( "1. fail" );
  }

  const pid_t pid = fork();
  if ( pid == 0 ) {
    // I'm the child

    // make the child realtime prio, but less than the parent
    sched_param param2;
    sched_getparam( 0, &param2 );
    param2.sched_priority = 1;
    if ( 0 != sched_setscheduler( 0, SCHED_FIFO, &param2 ) ) {
      perror( "2. fail" );
    } else {
      printf("running %s with SCHED_FIFO\n", argv[0]);
    }

    // drop privileges
    if ( geteuid() != getuid() ) {
      seteuid( getuid() );
      if ( geteuid() != getuid() ) {
        return -1;
      }

      execv( argv[0], argv );
      perror( argv[0] );
      return -1;
    }
    fprintf(stderr, "you need to make benchmark suid root\n");
    return -1;
  }

  timeval timeout;
  for ( int i = 0; i < 120; ++i ) {
    timeout.tv_sec =  1;
    timeout.tv_usec = 0;
    select( 0, 0, 0, 0, &timeout );
    if ( pid == waitpid( pid, 0, WNOHANG ) ) {
      return 0;
    }
  }

  printf("%s took too long (> 120s) and was killed\n", argv[0]);
  kill( pid, SIGKILL );

  return 0;
}
