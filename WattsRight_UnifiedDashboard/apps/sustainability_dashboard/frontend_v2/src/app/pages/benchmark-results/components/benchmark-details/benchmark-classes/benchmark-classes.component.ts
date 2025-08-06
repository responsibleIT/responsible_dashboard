import {Component, Input} from '@angular/core';
import {ClassificationData, ClassPerformance} from '@app/types/pruning.types';
import {
  BenchmarkClassComponent
} from '@app/pages/benchmark-results/components/benchmark-details/benchmark-classes/benchmark-class/benchmark-class.component';
import {NgForOf} from '@angular/common';

@Component({
  selector: 'app-benchmark-classes',
  imports: [
    BenchmarkClassComponent,
    NgForOf
  ],
  templateUrl: './benchmark-classes.component.html',
  styleUrl: './benchmark-classes.component.scss'
})
export class BenchmarkClassesComponent {

  @Input() classes: ClassPerformance[];

}
