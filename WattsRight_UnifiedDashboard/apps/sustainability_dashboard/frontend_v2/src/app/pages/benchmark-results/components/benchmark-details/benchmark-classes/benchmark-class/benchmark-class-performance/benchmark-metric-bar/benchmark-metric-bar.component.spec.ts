import { ComponentFixture, TestBed } from '@angular/core/testing';

import { BenchmarkMetricBarComponent } from './benchmark-metric-bar.component';

describe('BenchmarkMetricBarComponent', () => {
  let component: BenchmarkMetricBarComponent;
  let fixture: ComponentFixture<BenchmarkMetricBarComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [BenchmarkMetricBarComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(BenchmarkMetricBarComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
