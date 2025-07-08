import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkMetricCardComponent } from './benchmark-metric-card.component';

describe('BenchmarkMetricCardComponent', () => {
  let component: BenchmarkMetricCardComponent;
  let fixture: ComponentFixture<BenchmarkMetricCardComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BenchmarkMetricCardComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkMetricCardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
